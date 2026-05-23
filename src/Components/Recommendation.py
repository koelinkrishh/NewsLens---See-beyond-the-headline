import os
import sys
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import pandas as pd
import numpy as np
from typing import Optional, Union

from src.Components.Embedding_lc import ArticleEmbeddingEngine
from src.logger import logging
from src.exception import CustomException

class NewsRecommender:
    """
    Recommendation System for News Articles using a custom PostgreSQL schema and pgvector.
    Stores exact columns: id, article_text, summary_text, and vector embeddings.
    """
    def __init__(self, engine: Optional[ArticleEmbeddingEngine] = None):
        logging.info("Initializing Custom NewsRecommender...")
        try:
            # 1. Initialize LangChain Embedding Engine
            self.engine = engine or ArticleEmbeddingEngine(embedding_type="huggingface_local")
            
            # 2. Fetch PostgreSQL credentials
            db_user = os.getenv("POSTGRES_USER", "postgres")
            db_pass = os.getenv("POSTGRES_PASSWORD", "root")
            db_host = os.getenv("POSTGRES_HOST", "localhost")
            db_port = os.getenv("POSTGRES_PORT", "5432")
            db_name = os.getenv("POSTGRES_DB", "news_rag")
            
            if not (db_user and db_pass):
                raise ValueError("Postgres credentials (POSTGRES_USER, POSTGRES_PASSWORD) must be set in environment variables.")
                
            # 3. Connect to Database directly using psycopg2
            self.conn = psycopg2.connect(
                dbname=db_name,
                user=db_user,
                password=db_pass,
                host=db_host,
                port=db_port
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            
            # 4. Setup pgvector and Table Schema
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(self.conn)
            self._create_table()
            
            logging.info("Custom NewsRecommender initialized successfully.")
            
        except Exception as e:
            logging.error(f"Error initializing NewsRecommender: {e}")
            raise CustomException(e, sys)

    def _create_table(self):
        try:
            dim = getattr(self.engine, 'dimension', 384)
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS article_recommendations (
                id SERIAL PRIMARY KEY,
                article_text TEXT NOT NULL,
                summary_text TEXT,
                embedding vector({dim})
            );
            """
            self.cursor.execute(create_table_query)
        except Exception as e:
            logging.error(f"Failed to create article_recommendations table: {e}")
            raise CustomException(e, sys)

    def ingest_dataset(self, df: pd.DataFrame, batch_size: int = 100):
        """
        Embeds and ingests the entire dataset into PostgreSQL table permanently.
        Skips ingestion completely if the table already contains data.
        """
        try:
            # Check if table is populated with at least 100 records
            self.cursor.execute("SELECT COUNT(*) FROM article_recommendations;")
            count = self.cursor.fetchone()[0]
            if count > 100:
                logging.info(f"Database already populated with {count} articles. Skipping ingestion.")
                return
            
            logging.info(f"Ingesting {len(df)} articles into recommendation database...")
            
            # Ensure required columns
            if 'Content' not in df.columns:
                raise ValueError("DataFrame must contain a 'Content' column.")
            
            has_summary = 'Summary' in df.columns
            has_embedding = 'embedding' in df.columns
            
            texts = df['Content'].tolist()
            summaries = df['Summary'].tolist() if has_summary else [None] * len(texts)
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_summaries = summaries[i:i+batch_size]
                
                if has_embedding:
                    # Load pre-computed embeddings and convert to list of floats if they are numpy arrays
                    batch_embeddings = [
                        emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                        for emb in df['embedding'].iloc[i:i+batch_size]
                    ]
                else:
                    # Compute embeddings using LangChain engine
                    # ArticleEmbeddingEngine returns a numpy array
                    raw_embeddings = self.engine.embed_chunks(batch_texts)
                    batch_embeddings = [emb.tolist() for emb in raw_embeddings]
                
                records = []
                for text, summary, emb in zip(batch_texts, batch_summaries, batch_embeddings):
                    records.append((text, summary, emb))
                
                insert_query = """
                INSERT INTO article_recommendations (article_text, summary_text, embedding)
                VALUES %s;
                """
                execute_values(self.cursor, insert_query, records)
                logging.info(f"Ingested batch {i // batch_size + 1} (Total ingested: {min(i+batch_size, len(texts))})")
                
            logging.info("Dataset ingestion completed successfully.")
            
        except Exception as e:
            logging.error(f"Error during dataset ingestion: {e}")
            raise CustomException(e, sys)

    def recommend_similar(self, query: Union[str, np.ndarray, list], top_k: int = 3) -> pd.DataFrame:
        """
        Retrieves top_k similar articles based on a text query or a direct vector embedding.
        Returns the entire row representation as a pandas DataFrame.
        """
        try:
            # 1. Resolve query into an embedding vector
            if isinstance(query, str):
                # Text query: generate embedding using the engine
                embedding_vector = self.engine.embed_query(query).tolist()
            elif isinstance(query, np.ndarray):
                embedding_vector = query.tolist()
            elif isinstance(query, list):
                embedding_vector = query
            else:
                raise ValueError("Query must be a string, numpy array, or list representing the vector.")
            
            # 2. Execute pgvector search
            search_query = """
            SELECT id, article_text, summary_text, 1 - (embedding <=> %s::vector) AS similarity_score
            FROM article_recommendations
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """
            self.cursor.execute(search_query, (embedding_vector, embedding_vector, top_k))
            results = self.cursor.fetchall()
            
            # 3. Format output as a pandas DataFrame
            columns = ["id", "article_text", "summary_text", "similarity_score"]
            df_results = pd.DataFrame(results, columns=columns)
            
            return df_results
            
        except Exception as e:
            logging.error(f"Error finding recommendations: {e}")
            raise CustomException(e, sys)
            
    def close(self):
        """Clean up database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.config import CLEANED_DATASET_PARQUET
    
    print("========================================")
    print("Testing News Recommendation System (Custom psycopg2 & pgvector)...")
    print("========================================")
    
    # Load env keys for Postgres
    load_dotenv()
    
    try:
        # 1. Initialize Recommender and Embedding Engine
        print("\n1. Initializing Embedding Engine and NewsRecommender...")
        recommender = NewsRecommender()
        
        # 2. Verify dataset exists
        print(f"\n2. Verifying Dataset against Database...")
        if not os.path.exists(CLEANED_DATASET_PARQUET):
            print(f"[ERROR] Parquet file not found at {CLEANED_DATASET_PARQUET}")
            sys.exit(1)
            
        df = pd.read_parquet(CLEANED_DATASET_PARQUET)
        
        # 3. Ingest Entire Dataset (Recommender handles > 100 skip logic)
        print(f"\n3. Ingesting articles into PostgreSQL vector table...")
        recommender.ingest_dataset(df, batch_size=200)
        
        # 4. Test Recommendation
        print("\n4. Running Inference using Text Query...")
        test_article = "Apple has unveiled its latest iteration of the iPhone, featuring a newly designed titanium chassis and the powerful A17 Pro chip. The new device boasts significant improvements in camera performance and gaming capabilities, solidifying its position in the premium smartphone market."
        print(f"\nQuery Article Snippet:\n'{test_article}'")
        
        # Inference using pure string text
        df_recommendations = recommender.recommend_similar(test_article, top_k=3)
        
        print("\n================ RECOMMENDATIONS ================")
        print("Returned Format: pandas.DataFrame")
        print(df_recommendations[['id', 'similarity_score', 'summary_text']].to_string(index=False))
        print("=================================================\n")
        
        print("\n[SUCCESS] RECOMMENDATION SYSTEM END-TO-END VERIFICATION COMPLETED SUCCESSFULLY!")
        
        recommender.close()
        
    except Exception as e:
        print(f"\n[ERROR] Test run encountered an issue: {e}")
        import traceback
        traceback.print_exc()

