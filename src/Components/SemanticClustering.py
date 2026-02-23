import os
import sys
import joblib
import numpy as np
import pandas as pd
import spacy
from typing import List

# Loading model
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Local Imports
from src.config import *
from src.logger import logging
from src.exception import CustomException

"""
Refinement: TF-IDF will be dominated with global tern frequency, resulting in similar cluster labels
which is not ideally, So, Instead we switch to c-TF-IDF
"""

class KMeansTopicLabeler:
    """
    KMeans Topic Labeler Class
    """
    Invalid_terms = {"mr", "mrs", "ms", "dr", "said", "say", "says", "saying", "told", "according"}
    
    def __init__(self, n_clusters=10, spacy_model:str = SPACY_MODEL,):
        logging.info("Initializing KMeans Topic Labeler...")
        
        try:
            self.n_cluster = n_clusters
            self.Kmeans = MiniBatchKMeans(n_clusters=self.n_cluster, random_state=42, n_init='auto')
            self.nlp = spacy.load(spacy_model)
            
            # fitted after training
            self.vectorizer = None
            self.cluster_labels = None
            
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def is_valid_topic_term(self, term:str) -> bool:
        if not term.isalpha():
            return False
        if term.lower() in self.Invalid_terms:
            return False
        if len(term) <= 2:
            return False
        
        doc = self.nlp(term)
        for token in doc:
            # reject verbs, auxiliaries, pronouns
            if token.pos_ in {"VERB", "AUX", "PRON"}:
                return False
        return True
    
    def Label_clusters(self, df:pd.DataFrame, text_col:str="Content", emb_col:str="embedding",
        max_df:float=0.75, min_df:int=3, candidate_pool=25, top_n:int=10) -> dict:
        logging.info(f"Running KMeans Clustering (k={self.n_cluster})...")
        
        try:
            embeddings = np.vstack(df[emb_col].values)
            df["cluster"] = self.Kmeans.fit_predict(embeddings).astype(str)
            
            logging.info("Extraction TF-IDF labels per cluster")
            doc_per_cluster = df.groupby("cluster")[text_col].apply(lambda x: " ".join(x)).reset_index()
            
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df, max_df=max_df)
            cluster_tfidf = self.vectorizer.fit_transform(doc_per_cluster[text_col])
            vocab = self.vectorizer.get_feature_names_out()
            
            self.cluster_labels = {}
            for i, cluster_id in enumerate(doc_per_cluster["cluster"]):
                # Get scores for this specific cluster row
                scores = cluster_tfidf[i].toarray().flatten()
                
                # Take more candidates than needed, then filter linguistically
                sorted_idx = np.argsort(scores)[::-1][:candidate_pool]
                sorted_terms = [vocab[idx] for idx in sorted_idx if scores[idx] > 0]
                
                actual_terms = []
                for term in sorted_terms:
                    if self.is_valid_topic_term(term):
                        actual_terms.append(term)
                    if len(actual_terms)==top_n: break
                    
                self.cluster_labels[cluster_id] = actual_terms
            
            logging.info("KMeans Clustering & Labeling Complete")
            return df, self.cluster_labels
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def save(self):
        joblib.dump(self.Kmeans, KMEANS_MODEL_DIR)
        joblib.dump(self.vectorizer, KMEANS_VECTORIZER)
        joblib.dump(self.cluster_labels, KMEANS_LABELS)
        
        logging.info("Success! Saved to disk")
        
    def load(self):
        self.Kmeans = joblib.load(KMEANS_MODEL_DIR)
        self.vectorizer = joblib.load(KMEANS_VECTORIZER)
        self.cluster_labels = joblib.load(KMEANS_LABELS)
        
        logging.info("Loaded from disk")
        
    # INFERENCE
    def predict(self, text:str|List[str], embedding:np.ndarray):
        """ Cluster label for a given text and get its keywords. """
        if not hasattr(self.Kmeans, "cluster_centers_"):
            raise ValueError("Model is not trained or loaded")
        if isinstance(embedding, dict):
            raise ValueError("Embedding must be a numeric vector, not model output dict")

        # reshape embedding to 2D
        emb = np.array(embedding).astype(np.float32).reshape(1, -1)

        # predict cluster
        cluster_id = str(self.Kmeans.predict(emb)[0])

        # get labels
        if hasattr(self, "cluster_labels"):
            labels = self.cluster_labels.get(cluster_id, [])
        else:
            labels = []

        return {
            "text": text,
            "cluster": cluster_id,
            "labels": labels,
        }
        
        
                
# ---------------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------------        
if __name__ == "__main__":
    try:
        df = pd.read_parquet(EMBEDDED_DATASET)
        
        label_model = KMeansTopicLabeler(n_clusters=10)
        dataset, labels = label_model.Label_clusters(df=df)
        # print("Labels: ", labels)
        # print("Dataset: ", dataset)
        
        sample = df.sample(1)
        query = sample["Content"].values[0]
        emm = sample["embedding"].values[0]
        
        print("Embedding: ", emm.shape)
        print("Query: ", query[:100])
        print(label_model.predict(query, embedding=emm))
        
        label_model.save()
        
        logging.info("✅ Success! Saved to disk")
        # Saving dataframe
        df.to_parquet(CLUSTER_DATASET, index=False)
        logging.info("Successfully saved dataframe with clusters from KMeans Clustering")
        
    except Exception as e:
        logging.error("Error during KMeans Clustering pipeline.")
        raise CustomException(e, sys)
        
        
