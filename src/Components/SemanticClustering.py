import os
import sys
import joblib
import numpy as np
import pandas as pd
import spacy

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
            
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df, max_df=max_df)
            cluster_tfidf = vectorizer.fit_transform(doc_per_cluster[text_col])
            vocab = vectorizer.get_feature_names_out()
            
            cluster_labels = {}
            for i, cluster_id in enumerate(df["cluster"].unique()):
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
                    
                cluster_labels[cluster_id] = actual_terms
            
            logging.info("KMeans Clustering & Labeling Complete")
            return df, cluster_labels
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
                
# ---------------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------------        
if __name__ == "__main__":
    try:
        df = pd.read_parquet(EMBEDDED_DATASET)
        
        
        label_model = KMeansTopicLabeler(n_clusters=10)
        dataset, labels = label_model.Label_clusters(df=df)
        print("Labels: ", labels)
        print("Dataset: ", dataset)
        
        # Save artifacts
        joblib.dump(label_model, KMEANS_MODEL_DIR)
        joblib.dump(labels, KMEANS_LABELS)
        
        logging.info("✅ Success! Saved to disk")
        # Saving dataframe
        df.to_parquet(CLUSTER_DATASET, index=False)
        logging.info("Successfully saved dataframe with clusters from KMeans Clustering")
        
    except Exception as e:
        logging.error("Error during KMeans Clustering pipeline.")
        raise CustomException(e, sys)
        
        
