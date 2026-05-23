import os
import sys
import hashlib
import json
from typing import List, Optional, Dict, Any, Literal

# LangChain Imports
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector

# LangChain LLM Provider Imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Core system imports
from src.Components.Embedding_lc import ArticleEmbeddingEngine  # Fixed: was Embedding.py
from src.Components.utils import clean_text
from src.config import FAISS_RAG_INDEX, RAG_REGISTRY_JSON
from src.logger import logging
from src.exception import CustomException


class LCEmbeddingsAdapter(Embeddings):
    """
    Thin LangChain Embeddings adapter around ArticleEmbeddingEngine from Embedding_lc.
    FAISS and PGVector vectorstores require the LangChain Embeddings interface.
    """
    def __init__(self, engine: ArticleEmbeddingEngine):
        self.engine = engine
        logging.info("LCEmbeddingsAdapter initialized.")

    def embed_documents(self, texts: list) -> list:
        if not texts:
            return []
        return self.engine.embed_chunks(texts).tolist()

    def embed_query(self, text: str) -> list:
        return self.engine.embed_query(text).tolist()


class RAG:
    """
    High-Performance Conversational RAG Engine.
    Uses either FAISS or PGVector as explicitly requested.
    """
    def __init__(self, engine: Optional[ArticleEmbeddingEngine] = None, db_type: Literal["FAISS", "pgvector"] = "pgvector"):
        """ Initializes the RAG engine with the specified vector database type and embedding engine.
        
        Args:
            engine (Optional[ArticleEmbeddingEngine]): An instance of the ArticleEmbeddingEngine to use for embeddings. If None, a default instance will be created.
            db_type (Literal["FAISS", "pgvector"]): The type of vector database to use for retrieval. Must be either "FAISS" or "pgvector"
        """
        try:
            logging.info(f"Initializing RAG Engine with {db_type}...")
            
            self.db_type = db_type
            self.faiss_path = FAISS_RAG_INDEX
            self.registry_path = RAG_REGISTRY_JSON
            
            # Load processed articles registry to prevent duplicate work
            self.registry = self._load_registry()
            
            # Initialize LC Embedding Engine and wrap it in the LangChain Embeddings adapter
            self.engine = engine or ArticleEmbeddingEngine()
            self.embeddings = LCEmbeddingsAdapter(self.engine)
            
            self.vectorstore = None
            
            if self.db_type == "pgvector":
                db_user = os.getenv("POSTGRES_USER", "postgres")
                db_pass = os.getenv("POSTGRES_PASSWORD", "root")
                db_host = os.getenv("POSTGRES_HOST", "localhost")
                db_port = os.getenv("POSTGRES_PORT", "5432")
                db_name = os.getenv("POSTGRES_DB", "news_rag")
                
                if not (db_user and db_pass):
                    raise ValueError("Postgres credentials (POSTGRES_USER, POSTGRES_PASSWORD) must be set in environment variables to use pgvector.")
                
                self.connection_string = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
                logging.info(f"Attempting to connect to PostgreSQL/pgvector database '{db_name}' on {db_host}:{db_port}...")
                
                try:
                    self.vectorstore = PGVector(
                        self.embeddings,
                        connection=self.connection_string,
                        embedding_length=self.engine.dimension,
                        collection_name="news_chunks"
                    )
                    logging.info("Successfully connected to PostgreSQL and initialized PGVector.")
                except Exception as e:
                    logging.error(
                        "PostgreSQL/pgvector initialization failed. "
                        "Verify POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, "
                        "and that the database permits table creation."
                    )
                    logging.error(f"PGVector error: {e}")
                    raise CustomException(
                        ValueError(
                            "PGVector initialization failed. Ensure your Postgres/pgvector setup is correct and that the installed langchain_postgres version supports the current constructor signature."
                        ),
                        sys
                    )
                
            elif self.db_type == "FAISS":
                logging.info("Initializing FAISS local vector store...")
                if os.path.exists(self.faiss_path):
                    try:
                        self.vectorstore = FAISS.load_local(
                            self.faiss_path, self.embeddings, allow_dangerous_deserialization=True
                        )
                        logging.info("Successfully loaded existing local FAISS RAG index.")
                    except Exception as e:
                        logging.warning(f"Failed to load local FAISS index ({e}). It will be recreated.")
                        self.vectorstore = None
                else:
                    self.vectorstore = None
                    logging.info("No local FAISS RAG index found. It will be initialized on first ingestion.")
            else:
                raise ValueError(f"Invalid db_type: {self.db_type}. Choose 'FAISS' or 'pgvector'.")
                    
        except Exception as e:
            logging.error(f"Error during RAG Engine initialization: {e}")
            raise CustomException(e, sys)

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load RAG registry file: {e}")
        return {"processed_hashes": []}

    def _save_registry(self):
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save RAG registry file: {e}")

    def add_article(self, article: str) -> str:
        """
        Chunks, embeds, and stores article chunks in the vector database.
        Checks SHA-256 hash to prevent duplicate embedding work.
        """
        try:
            if not article or not article.strip():
                logging.warning("Received empty article text for ingestion.")
                return ""
                
            # Generate unique hash for deduplication
            article_hash = hashlib.sha256(article.encode('utf-8')).hexdigest()
            
            if article_hash in self.registry.get("processed_hashes", []):
                logging.info(f"Article {article_hash[:8]}... already exists in vector store. Skipping ingestion.")
                return article_hash

            # Chunk using the LC engine's built-in chunker (RecursiveCharacterTextSplitter)
            logging.info(f"Processing new article {article_hash[:8]}... for ingestion...")
            chunks = self.engine.chunk_article(article)
            
            if not chunks:
                logging.warning("No meaningful chunks could be extracted from article.")
                return ""
                
            # Create LangChain Documents with article hash metadata for filtering
            docs = [Document(page_content=chunk, metadata={"article_hash": article_hash}) for chunk in chunks]
            
            if self.db_type == "pgvector":
                # Add to PGVector
                self.vectorstore.add_documents(docs)
            elif self.db_type == "FAISS":
                # Add to FAISS and save locally
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_documents(docs, self.embeddings)
                else:
                    self.vectorstore.add_documents(docs)
                self.vectorstore.save_local(self.faiss_path)
                
            # Register hash
            if "processed_hashes" not in self.registry:
                self.registry["processed_hashes"] = []
            self.registry["processed_hashes"].append(article_hash)
            self._save_registry()
            
            logging.info(f"Successfully processed and stored {len(chunks)} chunks for article (hash: {article_hash[:8]}).")
            return article_hash
            
        except Exception as e:
            logging.error(f"Failed to ingest article: {e}")
            raise CustomException(e, sys)

    def retrieve(self, query: str, article_hash: str, k: int = 4) -> List[Document]:
        """
        Retrieves top k chunks matching the query, filtered strictly by the current article_hash.
        """
        try:
            if self.vectorstore is None:
                logging.warning("RAG retrieve called but no vectorstore exists.")
                return []
                
            # Search with metadata filtering to prevent cross-article chunk leakages
            results = self.vectorstore.similarity_search(
                query, k=k, filter={"article_hash": article_hash}
            )
            return results
        except Exception as e:
            logging.error(f"Failed to retrieve contexts: {e}")
            raise CustomException(e, sys)


class QA:
    """
    High-Fidelity Conversational Q&A Engine.
    Uses retrieved context chunks from RAG to answer queries with precise context grounding.
    """
    def __init__(self,
        rag: Optional[RAG] = None,
        provider: str = "groq",
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None
    ):
        try:
            logging.info("Initializing QA Engine...")
            self.rag = rag or RAG()
            self.provider = provider.lower()
            self.model = model
            
            # Setup LLM based on provider parameters (matches NER_lc.py parity)
            if self.provider == "groq":
                key = api_key or os.getenv("GROQ_API_KEY")
                if not key:
                    raise ValueError("GROQ_API_KEY is missing. You must provide a valid Groq API Key.")
                self.llm = ChatGroq(model=self.model, groq_api_key=key, temperature=0)

            elif self.provider == "openai":
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError("OPENAI_API_KEY is missing. You must provide a valid OpenAI API Key.")
                self.llm = ChatOpenAI(model=self.model, openai_api_key=key, temperature=0)

            elif self.provider == "huggingface":
                key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not key:
                    raise ValueError("HF_TOKEN / HUGGINGFACEHUB_API_TOKEN is missing. You must provide a valid Hugging Face Token.")
                self.llm = ChatOpenAI(
                    model=self.model,
                    openai_api_key=key,
                    base_url="https://router.huggingface.co/v1",
                    temperature=0
                )
            else:
                raise ValueError(f"Unsupported provider: '{self.provider}'. Choose from 'groq', 'openai', or 'huggingface'.")

            # Initialize grounding system prompt
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant answering questions about a news article.
                
                Use the following pieces of retrieved context from the article to answer the question.
                If you don't know the answer or if the context doesn't contain the answer, say that you cannot find the answer in the article. Do not make up or assume any details.
                
                Context:
                {context}"""),
                ("human", "Question: {question}")
            ])
            logging.info("QA Engine initialized successfully.")
            
        except Exception as e:
            logging.error(f"Failed to initialize QA Engine: {e}")
            raise CustomException(e, sys)
        
    def answer_question(self, article: str, question: str) -> Dict[str, Any]:
        """
        Runs the full end-to-end RAG QA pipeline:
        1. Ingest article (skips automatically if hash exists).
        2. Retrieves relevant grounded context chunks.
        3. Invokes the LLM to get the answer.
        """
        try:
            if not article or not article.strip():
                return {"answer": "Error: No article content provided.", "context": []}
            if not question or not question.strip():
                return {"answer": "Error: Please provide a question.", "context": []}

            # 1. Ingest/retrieve article hash
            article_hash = self.rag.add_article(article)
            if not article_hash:
                return {"answer": "Error: Failed to process article for QA.", "context": []}
            
            # 2. Retrieve top chunks
            retrieved_docs = self.rag.retrieve(question, article_hash, k=4)
            context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # 3. Get grounded response from LLM
            formatted_messages = self.prompt_template.format_messages(
                context=context_str,
                question=question
            )
            response = self.llm.invoke(formatted_messages)
            answer = response.content
            
            return {
                "answer": answer,
                "context": [doc.page_content for doc in retrieved_docs]
            }
            
        except Exception as e:
            logging.error(f"QA pipeline invocation failed: {e}")
            raise CustomException(e, sys)
