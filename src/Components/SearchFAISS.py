import os
import sys
import faiss
import numpy as np
import pandas as pd
from typing import List

from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# --- Local Imports ---
from src.config import *
from src.logger import logging
from src.exception import CustomException

# Chaching for faster processing
from functools import lru_cache


class SemanticSearchEngine:
    """
    High Performance Routed Semantic Search
    
    Instead of searching all embeddings, we only search embeddings that are relevant to the query
    1. Predicts the topic of the search query
    2. Routes the query to a spacifc FAISS 'shard' (index) containing only that topic
    3. Falls back to a global index if the topic is uncertain or bypassed.
    """
    def __init__(self, topic_model:BERTopic, df:pd.DataFrame, embedding_col:str="embedding",topic_col: str = "Topic"):
        logging.info("Initializing FAISS Search Engine...")
        try:
            self.topic_model = topic_model
            self.df = df
            self.embedding_col = embedding_col
            self.topic_col = topic_col
            
            self.embedding = np.vstack(df[embedding_col].values)
            self.em_dim = self.embedding.shape[1]
            
            # 1. Normalize for Cosine Similarity (Inner Product in FAISS)
            self.embedding = self._normalize(self.embedding)
            
            self.indices = {}   # Dict to store FAISS index per topic
            self.id_maps = {}   # Dict to store mapping from FAISS ID -> Dataframe index
            
            self._build_indices(df, topic_col=topic_col)
            logging.info("FAISS Search Engine initialized successfully.")
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def _normalize(self, v:np.ndarray) -> np.ndarray:
        if v.ndim == 1:
            v = v.reshape(1, -1)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        div = np.where(norm==0,1,norm)
        return (v/div).astype("float32")

    def _build_indices(self, df:pd.DataFrame, topic_col:str):
        """
        Builds a seperate IndexFlatIP for each topic cluster
        """
        try:
            unique_topics = df[topic_col].unique()
            logging.info(f"Building FAISS indices for {len(unique_topics)} topics.")
            
            for topic_id in unique_topics:
                # 1. Boolean mask for topic match
                mask = (df[topic_col]==topic_id).values
                # 2. Specific embedding for this cluster
                subset_embedding = self.embedding[mask]
                
                # 3. Store the original DataFrame indices to map back
                self.id_maps[topic_id] = df.index[mask].values
                # 4. Create FAISS index [Inner Product]
                idx = faiss.IndexFlatIP(self.em_dim)
                idx.add(subset_embedding)
                self.indices[topic_id] = idx
                
            self.id_maps[-1] = df.index.values
            global_idx = faiss.IndexFlatIP(self.em_dim)
            global_idx.add(self.embedding)
            self.indices[-1] = global_idx
            
        except Exception as e:
            raise CustomException(e, sys)
    
    @lru_cache(maxsize=16) # Store the result for last 64 recommended articles
    def search(self, query:str, top_k:int=5, filter=False) -> pd.DataFrame:
        """
        Advance semantic search function with Topic Routing
    
        Args:
            query (str | List[str]): Query Article or list of queries
            top_k (int, optional): No. of results to return.
            filter_space (bool, optional): Filter search space based on Topic Cluster matching.

        Returns:
            pd.DataFrame: Top K similar articles
        """
        try:
            # 1. Embed and Normalize Query
            embedder = self.topic_model.embedding_model
            query_vec = embedder.embed([query])
            query_vec = self._normalize(query_vec).astype("float32")
            
            # 2. Perfic Topic and Routing Logic
            target_topic = -1
            if filter:
                pred_topics, _ = self.topic_model.transform([query])
                target_topic = pred_topics[0]
            ## Fallback to global search if the topic isn't in our indices
            if target_topic not in self.indices:
                target_topic = -1
            
            # 3. Execute FAISS search
            index = self.indices[target_topic]
            top_k = min(top_k, index.ntotal) # safety limit
            
            scores_matrix, idx_matrix = index.search(query_vec, top_k)
            
            # 4. Map results index back to DataFrame
            faiss_idx = idx_matrix[0]
            scores = scores_matrix[0]
            df_idx = self.id_maps[target_topic][faiss_idx]
            
            # 5. Construct Final Results
            results_df = self.df.loc[df_idx].copy()
            results_df["Search Score"] = scores
            
            return results_df, target_topic
        except Exception as e:
            logging.error(f"Search failed for query: '{query[:30]}...'")
            raise CustomException(e, sys)


# ---------------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        df = pd.read_parquet(FINAL_DATASET_PARQUET)
        
        topic_model = BERTopic.load(BERTOPIC_MODEL_PARAMETERS, embedding_model=SENTENCE_TRANSFORMER_MODEL)
        search_engine = SemanticSearchEngine(topic_model, df)
        
        test_query = df.loc[100,'Content']
        results, topics = search_engine.search(test_query, filter=True, top_k=3)
        
        print("Test Query: ", test_query)
        
        print(f"\n Successfully found topic: {topics}")
        print("- x -"*20)
        print(results[["Content", "Summary", "cluster", "Topic", "Search Score" ]])
        
    except Exception as e:
        logging.error(f"Search Engine Test Failed: {e}")
        raise CustomException(e, sys)  
        

