from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import time
import sys
import os

# Ensure src in path -> Add root project file path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import *
from src.Components.FeatureGeneration import TextFeatureGenerator
from src.Components.Embedding import ArticleEmbeddingEngine
from src.Components.SemanticClustering import KMeansTopicLabeler
from src.Components.SemanticTopicModel import SemanticTopicModel
from src.Components.Summarization import NewsSummarizer
from src.Components.NER import InformationExtractor
from src.Components.SearchFAISS import SemanticSearchEngine
from bertopic import BERTopic

app = FastAPI(title="News Intelligence Platform API")

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins like ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Security Setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate API Key")

# 3. Simple In-Memory Rate Limiter
request_counts = {} # simple dict to track requests per client IP

# def check_rate_limit(client_ip: str):
#     now = time.time()
#     if client_ip not in request_counts:
#         request_counts[client_ip] = []
    
#     # Clean up old requests
#     request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < RATE_LIMIT_WINDOW]
#     """ BRAIN for rate-limiter
#     This is using a sliding window to maintain timestamps of every requests made by user.
#     If (current_time-last_request_time) < RATE_LIMIT_WINDOW, then remove old requests.
#     """
    
#     if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
#         raise HTTPException(status_code=429, detail="Too many requests")
    
#     request_counts[client_ip].append(now)

models = {}

# API health check -> Needed for deployment platform    
@app.get("/")
def health_check():
    return {"status": "online", "message": "News Intelligence API is running"}

# Automatic work done through api before starting app
@app.on_event("startup")
def load_models():
    models['feature_gen'] = TextFeatureGenerator()
    models['embedder'] = ArticleEmbeddingEngine()
    
    kmeans = KMeansTopicLabeler()
    kmeans.load()
    models['kmeans'] = kmeans
    
    topic_model = BERTopic.load(BERTOPIC_MODEL_PARAMETERS, embedding_model=models['embedder']) # """New every time: SENTENCE_TRANSFORMER_MODEL"""
    models['topic_model'] = topic_model
    
    models['summarizer'] = NewsSummarizer()
    models['ner'] = InformationExtractor()
    
    try:
        df = pd.read_parquet(FINAL_DATASET_PARQUET)
        models['search_engine'] = SemanticSearchEngine(topic_model, df)
    except Exception as e:
        models['search_engine'] = None

class ArticleRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)

class SummarizeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    compression: float = Field(0.5, ge=0.1, le=0.9)

# PRocessing task done on Inference time
@app.post("/process")
def process_article(req: ArticleRequest, api_key: APIKey = Depends(get_api_key)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    
    feat_df = models['feature_gen'].Create_features(req.text)
    emb = models['embedder'].create_embedding(req.text)
    
    # Return features as dict and embedding as list
    feat_dict = feat_df.iloc[0].to_dict()
    return {"features": feat_dict, "embedding": emb.tolist()}

# Summarization task 
@app.post("/summarize")
def summarize_article(req: SummarizeRequest, api_key: APIKey = Depends(get_api_key)):
    summary_text = models['summarizer'].summarize(req.text, compression=req.compression)
    return {"summary": summary_text}

# Clustering task
@app.post("/cluster")
def cluster_article(req: ArticleRequest, api_key: APIKey = Depends(get_api_key)):
    emb = models['embedder'].create_embedding(req.text)
    # KMeans
    kmeans_info = models['kmeans'].predict(req.text, emb)
    # BERTopic
    predicted_topic, prob = models['topic_model'].transform([req.text], np.array([emb]))
    topic_id = int(predicted_topic[0])
    topic_prob = prob[0].tolist() if prob is not None else []
    
    topic_keywords = models['topic_model'].get_topic(topic_id)
    if topic_keywords is False or topic_keywords is None:
        topic_keywords = []

    return {
        "kmeans": {
            "cluster_id": int(kmeans_info["cluster"]),
            "labels": kmeans_info.get("labels", [])
        },
        "bertopic": {
            "topic_id": topic_id,
            "topic_prob": topic_prob,
            "topic_keywords": topic_keywords
        }
    }

# NER task
@app.post("/ner")
def extract_ner(req: ArticleRequest, api_key: APIKey = Depends(get_api_key)):
    # Process text for spacy, gliner, and keywords
    ner_res = models['ner'].process_articles(req.text)
    
    keywords = ner_res['keywords']
    spacy_ents = ner_res['predefined_ents']
    gliner_ents = ner_res['custom_ents']
    
    # We can serialize simplified formats or just return HTMLs for visualization
    spacy_html = models['ner'].Visualize(req.text, type="spacy")
    gliner_html = models['ner'].Visualize(req.text, type="gliner", entities=gliner_ents)
    
    return {
        "spacy_html": spacy_html,
        "gliner_html": gliner_html,
        "spacy_ents": spacy_ents,
        "gliner_ents": gliner_ents,
        "keywords": keywords,
        "raw_ner_res": {
            "Keywords": keywords,
            "predefined_ent": spacy_ents,
            "custom_ent": gliner_ents
        }
    }

# Search task
@app.post("/search")
def search_similar(req: ArticleRequest, api_key: APIKey = Depends(get_api_key)):
    if models['search_engine'] is None:
        raise HTTPException(status_code=503, detail="Search engine offline")
    
    results, topic = models['search_engine'].search(req.text, top_k=5, filter=True)
    
    res_list = []
    if not results.empty:
        for idx, row in results.iterrows():
            res_dict = {
                "idx": idx,
                "Score": float(row.get("Search Score", 0.0)),
                "Title": str(row.get("Title", "")),
                "Content": str(row.get("Content", "")),
                "cluster": str(row.get("cluster", ""))
            }
            res_list.append(res_dict)
            
    return {"topic": int(topic), "results": res_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

