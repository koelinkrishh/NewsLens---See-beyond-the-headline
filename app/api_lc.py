"""
api_lc.py — LangChain-powered FastAPI Backend for NewLens
Endpoints wired to the 6 updated LC components.
"""

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.config import (
    API_KEY, MAX_TEXT_LENGTH,
    FAISS_RAG_INDEX, RAG_REGISTRY_JSON,
    CLEANED_DATASET_PARQUET, API_PORT
)
from src.Components.FeatureGeneration_lc import TextFeatureGenerator
from src.Components.Embedding_lc import ArticleEmbeddingEngine
from src.Components.NER_lc import InformationExtractor
from src.Components.QAchain import RAG, QA
from src.Components.Recommendation import NewsRecommender
from src.Components.Summarization_lc import NewsSummarizerLC
from src.logger import logging

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NewLens LC API",
    description="LangChain-based News Intelligence Backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Invalid API Key")

# ---------------------------------------------------------------------------
# Global model registry (loaded once at startup)
# ---------------------------------------------------------------------------
models = {}

@app.on_event("startup")
def load_models():
    logging.info("=== Starting NewLens LC API model loading ===")

    # Shared embedding engine (loaded once, reused by RAG and Recommender)
    logging.info("Loading ArticleEmbeddingEngine (huggingface_local)...")
    embedding_engine = ArticleEmbeddingEngine(embedding_type="huggingface_local")
    models["embedder"] = embedding_engine

    # Feature Generator — semantic off by default for speed (enable with provider="groq")
    logging.info("Loading TextFeatureGenerator (classical mode)...")
    models["feature_gen"] = TextFeatureGenerator(semantic=False)

    # NER — Groq provider, reads key from .env
    logging.info("Loading InformationExtractor (Groq)...")
    models["ner"] = InformationExtractor(provider="groq", model="llama-3.3-70b-versatile")

    # RAG — FAISS, shares the embedding engine
    logging.info("Loading RAG Engine ...")
    rag = RAG(engine=embedding_engine, db_type="pgvector")
    models["rag"] = rag

    # QA — shares the same RAG instance + Groq LLM
    logging.info("Loading QA Engine (Groq)...")
    models["qa"] = QA(rag=rag, provider="groq", model="llama-3.1-8b-instant")

    # Summarizer — Groq for speed
    logging.info("Loading NewsSummarizerLC (Groq)...")
    models["summarizer"] = NewsSummarizerLC(
        llm_type="groq",
        embedding_engine=embedding_engine  # Reuse the same engine — no double load
    )

    # Recommender — psycopg2 pgvector, shares embedding engine
    logging.info("Loading NewsRecommender (pgvector)...")
    try:
        recommender = NewsRecommender(engine=embedding_engine)
        # Auto-ingest dataset if the DB is empty
        from src.config import FINAL_DATASET_PARQUET
        # Use final dataset with pre-computed embeddings for instant loading if it exists, else fall back to cleaned dataset
        ingest_file = FINAL_DATASET_PARQUET if os.path.exists(FINAL_DATASET_PARQUET) else CLEANED_DATASET_PARQUET
        if os.path.exists(ingest_file):
            logging.info(f"Auto-ingesting recommendation dataset from: {ingest_file}...")
            df = pd.read_parquet(ingest_file)
            recommender.ingest_dataset(df, batch_size=1000)
        models["recommender"] = recommender
    except Exception as e:
        logging.error(f"Recommender startup failed (non-fatal): {e}")
        models["recommender"] = None

    logging.info("=== All models loaded successfully ===")


@app.on_event("shutdown")
def shutdown():
    if models.get("recommender"):
        models["recommender"].close()
        logging.info("Recommender DB connection closed.")

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class ArticleRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH, description="Full article text")

class SummarizeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    reduction_ratio: float = Field(0.5, ge=0.1, le=0.9, description="Target compression ratio (0.1=very short, 0.9=almost full length)")

class QARequest(BaseModel):
    article: str = Field(..., max_length=MAX_TEXT_LENGTH, description="Article the question is about")
    question: str = Field(..., max_length=500, description="Question to answer from the article")

class RecommendRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH, description="Article or query text")
    top_k: int = Field(5, ge=1, le=20, description="Number of similar articles to return")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "online", "api": "NewLens LC", "version": "2.0.0"}


@app.post("/process", tags=["Analysis"])
def process_article(req: ArticleRequest, api_key: APIKey = Depends(get_api_key)):
    """
    Extract heuristic text features (POS counts, readability, etc.)
    and compute the article embedding vector.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty article text")
    try:
        # Extract features (classical mode — fast, no LLM call)
        feat_df = models["feature_gen"].Create_features(req.text)
        feat_dict = feat_df.iloc[0].to_dict()

        # Embed the full article
        emb = models["embedder"].embed_query(req.text)

        return {
            "features": feat_dict,
            "embedding": emb.tolist(),
            "word_count": len(req.text.split()),
        }
    except Exception as e:
        logging.error(f"/process failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", tags=["Analysis"])
def summarize_article(req: SummarizeRequest, api_key: APIKey = Depends(get_api_key)):
    """
    Hybrid Extractive-Abstractive summarization using Groq LLM.
    reduction_ratio controls both the chunk selection and target output length.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty article text")
    try:
        summary = models["summarizer"].summarize(req.text, reduction_ratio=req.reduction_ratio)
        return {"summary": summary}
    except Exception as e:
        logging.error(f"/summarize failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ner", tags=["Analysis"])
def extract_ner(req: ArticleRequest, api_key: APIKey = Depends(get_api_key)):
    """
    Extract named entities (LLM-based via Groq + GLiNER label set) and
    semantic keywords (KeyBERT). Returns entities, highlighted HTML, and keywords.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty article text")
    try:
        ner_result = models["ner"].process_articles(req.text)

        entities = ner_result.get("entities", [])
        keywords = ner_result.get("keywords", [])
        predefined_ents = ner_result.get("predefined_ents", [])
        custom_ents = ner_result.get("custom_ents", [])

        # Generate highlighted HTML via NER_lc Visualize method
        html = models["ner"].Visualize(req.text, entities=entities)

        return {
            "html": html,
            "entities": entities,
            "predefined_ents": predefined_ents,
            "custom_ents": custom_ents,
            "keywords": keywords,
        }
    except Exception as e:
        logging.error(f"/ner failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa", tags=["QA"])
def answer_question(req: QARequest, api_key: APIKey = Depends(get_api_key)):
    """
    Retrieval-Augmented Generation (RAG) Q&A over the provided article.
    Chunks are stored in FAISS and retrieved by semantic similarity.
    Returns the LLM answer plus the exact context chunks used.
    """
    if not req.article.strip():
        raise HTTPException(status_code=400, detail="Empty article text")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    try:
        result = models["qa"].answer_question(req.article, req.question)
        return {
            "answer": result["answer"],
            "context_chunks": result["context"],
        }
    except Exception as e:
        logging.error(f"/qa failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", tags=["Recommendations"])
def recommend_articles(req: RecommendRequest, api_key: APIKey = Depends(get_api_key)):
    """
    Returns top_k similar articles from the pgvector database using cosine similarity.
    Query is embedded and matched against the article_recommendations table.
    """
    if models.get("recommender") is None:
        raise HTTPException(status_code=503, detail="Recommender offline — check PostgreSQL connection and dataset.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty query text")
    try:
        df_recs = models["recommender"].recommend_similar(req.text, top_k=req.top_k)
        # Convert to JSON-serializable list of dicts
        records = df_recs.to_dict(orient="records")
        return {"recommendations": records, "count": len(records)}
    except Exception as e:
        logging.error(f"/recommend failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
