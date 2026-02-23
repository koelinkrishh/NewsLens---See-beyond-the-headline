from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
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

models = {}

@app.get("/")
def health_check():
    return {"status": "online", "message": "News Intelligence API is running"}

@app.on_event("startup")
def load_models():
    models['feature_gen'] = TextFeatureGenerator()
    models['embedder'] = ArticleEmbeddingEngine()
    
    kmeans = KMeansTopicLabeler()
    kmeans.load()
    models['kmeans'] = kmeans
    
    topic_model = BERTopic.load(BERTOPIC_MODEL_PARAMETERS, embedding_model=SENTENCE_TRANSFORMER_MODEL)
    models['topic_model'] = topic_model
    
    models['summarizer'] = NewsSummarizer()
    models['ner'] = InformationExtractor()
    
    try:
        df = pd.read_parquet(FINAL_DATASET_PARQUET)
        models['search_engine'] = SemanticSearchEngine(topic_model, df)
    except Exception as e:
        models['search_engine'] = None

class ArticleRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    text: str
    compression: float = 0.5

@app.post("/process")
def process_article(req: ArticleRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    
    feat_df = models['feature_gen'].Create_features(req.text)
    emb = models['embedder'].create_embedding(req.text)
    
    # Return features as dict and embedding as list
    feat_dict = feat_df.iloc[0].to_dict()
    return {"features": feat_dict, "embedding": emb.tolist()}

@app.post("/summarize")
def summarize_article(req: SummarizeRequest):
    summary_text = models['summarizer'].summarize(req.text, compression=req.compression)
    return {"summary": summary_text}

@app.post("/cluster")
def cluster_article(req: ArticleRequest):
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

@app.post("/ner")
def extract_ner(req: ArticleRequest):
    # Process text for spacy, gliner, and keywords
    ner_res = models['ner'].process_articles(req.text)
    
    # We can serialize simplified formats or just return HTMLs for visualization
    spacy_html = models['ner'].Visualize(req.text, type="spacy")
    gliner_html = models['ner'].Visualize(req.text, type="gliner")
    
    # Extract entities and keywords
    spacy_ents = models['ner'].extract_spacy_entities(req.text)
    gliner_ents = models['ner'].extract_gliner_entities(req.text, labels=models['ner'].default_ner_labels)
    keywords = models['ner'].extract_keywords(req.text)
    
    return {
        "spacy_html": spacy_html,
        "gliner_html": gliner_html,
        "spacy_ents": spacy_ents,
        "gliner_ents": gliner_ents,
        "keywords": keywords,
        "raw_ner_res": ner_res
    }

@app.post("/search")
def search_similar(req: ArticleRequest):
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
    uvicorn.run(app, host="127.0.0.1", port=8000)

