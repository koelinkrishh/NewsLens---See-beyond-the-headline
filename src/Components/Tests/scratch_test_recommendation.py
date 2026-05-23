import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Components.Recommendation import NewsRecommender
from src.config import CLEANED_DATASET_PARQUET
from src.Components.Embedding_lc import ArticleEmbeddingEngine

if __name__ == "__main__":
    print("========================================")
    print("Testing News Recommendation System (Custom psycopg2 & pgvector)...")
    print("========================================")
    
    # Load env keys for Postgres
    load_dotenv()
    
    try:
        # 1. Initialize Recommender and Embedding Engine
        print("\n1. Initializing Embedding Engine and NewsRecommender...")
        engine = ArticleEmbeddingEngine(embedding_type="huggingface_local")
        recommender = NewsRecommender(engine=engine)
        
        # 2. Ingest Data (Skips automatically if already populated)
        print(f"\n2. Verifying Dataset against Database...")
        if not os.path.exists(CLEANED_DATASET_PARQUET):
            print(f"[ERROR] Parquet file not found at {CLEANED_DATASET_PARQUET}")
            sys.exit(1)
            
        df = pd.read_parquet(CLEANED_DATASET_PARQUET)
        
        # 3. Ingest Entire Dataset (Recommender handles skip logic)
        print(f"\n3. Ingesting {len(df)} articles into PostgreSQL vector table...")
        recommender.ingest_dataset(df, batch_size=200)
        
        # 4. Test Recommendation (Inference via Text Query)
        print("\n4. Running Inference using Text Query...")
        test_article = """Apple has unveiled its latest iteration of the iPhone, featuring a newly designed titanium chassis and the powerful A17 Pro chip. The new device boasts significant improvements in camera performance and gaming capabilities, solidifying its position in the premium smartphone market."""
        print(f"\nQuery Article Snippet:\n'{test_article}'")
        
        # Inference using pure string text
        df_recommendations = recommender.recommend_similar(test_article, top_k=3)
        
        print("\n================ RECOMMENDATIONS ================")
        print("Returned Format: pandas.DataFrame")
        print(df_recommendations[['id', 'similarity_score', 'summary_text']].to_string(index=False))
        print("=================================================\n")
        
        print("\n[SUCCESS] RECOMMENDATION SYSTEM END-TO-END VERIFICATION COMPLETED SUCCESSFULLY!")
        
        # Cleanup
        recommender.close()
        
    except Exception as e:
        print(f"\n[ERROR] Test run encountered an issue: {e}")
        import traceback
        traceback.print_exc()
