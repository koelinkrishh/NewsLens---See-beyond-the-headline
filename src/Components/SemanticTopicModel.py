''' 
Upto now, we have created components file for feature creation and model training.
Those code block needed to be run once for Setting up our Dataset.

those model only need to be loaded to create features at Inference.
The following components will also be trained once from dataset and use everyone at inference.
'''

import os
import sys
import joblib
import numpy as np
import pandas as pd
import umap
import hdbscan

# Loading model
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

# Local Imports
from src.config import *
from src.logger import logging
from src.exception import CustomException


class SemanticTopicModel:
    """
    BERTopic wrapper for training, inference, zero-shot cluster labeling,
    and semantic indexing.
    """
    def __init__(self, embedding_model_name: str = SENTENCE_TRANSFORMER_MODEL,):
        logging.info(f"Initializing Semantic Topic Model [{embedding_model_name}]...")
    
        try:    
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.umap_model = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0, metric='cosine', random_state=42)
            self.cluster_model = MiniBatchKMeans(n_clusters=20, random_state=42, n_init='auto')
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=10, max_df=0.8)

            self.topic_model = BERTopic(
                language="english",
                vectorizer_model=self.vectorizer,
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.cluster_model,
                calculate_probabilities=True,
                verbose=True, 
                nr_topics=None # let kmeans handle the count
            )
            logging.info("Semantic Topic Model initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)
        
    def fit_transform(self, df:pd.DataFrame, text_col:str="Content", emb_col:str="embedding") -> pd.DataFrame:
        logging.info("Fitting BERTopic pipeline)")
        
        try:
            texts = df[text_col].astype(str).tolist()
            embeddings = np.vstack(df[emb_col].values)

            topics, _ = self.topic_model.fit_transform(texts, embeddings)
            df['Topic'] = topics

            logging.info("BERTopic fit successfully.")
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_model(self):
        try:
            logging.info("Saving our BERTopic pipeline")
            # Saving standard BERTopic pipeline
            joblib.dump(self.topic_model, BERTOPIC_MODEL_DIR)
            # saving entire model:
            self.topic_model.save(BERTOPIC_MODEL_PARAMETERS, serialization="pytorch", save_embedding_model=True)

        except Exception as e:
            raise CustomException(e, sys)
        
    
# ---------------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------------     
if __name__ == "__main__":
    try:
        # We can load the kmeans output or the pure embeddings file
        logging.info(f"Loading data from {CLUSTER_DATASET}")
        df = pd.read_parquet(CLUSTER_DATASET)
        
        engine = SemanticTopicModel()
        dataset = engine.fit_transform(df)
        engine.save_model()
        
        # Save Final Dataset
        df.to_parquet(FINAL_DATASET_PARQUET, index=False)
        logging.info(f"✅ Success! Saved topic modeled data to {FINAL_DATASET_PARQUET}")

    except Exception as e:
        logging.error(f"Pipeline Failed: {e}")
        raise CustomException(e, sys)

