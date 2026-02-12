import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional

class SemanticTopicModel:
    """
    BERTopic wrapper for training, inference, zero-shot cluster labeling,
    and semantic indexing.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_df: int = 5,
        max_df: float = 0.75,
        ngram_range: tuple = (1, 2),
        calculate_probabilities: bool = True,
        verbose: bool = True,
        model_path: Optional[str] = None
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)

        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )

        if model_path:
            self.topic_model = BERTopic.load(
                model_path,
                embedding_model=self.embedding_model
            )
        else:
            self.topic_model = BERTopic(
                language="english",
                vectorizer_model=vectorizer,
                embedding_model=self.embedding_model,
                calculate_probabilities=calculate_probabilities,
                verbose=verbose
            )

        self.topic_labels: Dict[int, List[str]] = {}
        
    def fit(self, texts: List[str], embeddings: Optional[np.ndarray] = None, clusters: Optional[List[int]] = None):
        """
        Train BERTopic.
        If clusters are provided, they guide topic formation.
        """

        if embeddings is None:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        self.topics_, self.probs_ = self.topic_model.fit_transform(
            texts,
            embeddings,
            y=clusters
        )

        return self.topics_, self.probs_
    
    def label_clusters(self, top_n_words: int = 5) -> Dict[int, List[str]]:
        """
        Generate human-readable labels for each topic.
        """

        labels = {}
        for topic_id, words in self.topic_model.get_topics().items():
            if topic_id == -1:
                continue

            labels[topic_id] = [word for word, _ in words[:top_n_words]]

        self.topic_labels = labels
        return labels

    def transform(self, texts: List[str]):
        """
        Assign nearest topic to new documents at Inference
        """

        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        topics, probs = self.topic_model.transform(texts, embeddings)

        return topics, probs

    def add_semantic_topics(self, df: pd.DataFrame, text_col: str, topic_col: str = "topic", prob_col: str = "topic_confidence") -> pd.DataFrame:
        """
        Adds topic assignments + confidence scores to a dataframe.
        """

        texts = df[text_col].astype(str).tolist()
        topics, probs = self.transform(texts)

        df = df.copy()
        df[topic_col] = topics
        df[prob_col] = probs.max(axis=1)

        return df
    
    def get_topic_info(self, topic_id: int):
        """
        Get keywords + optional label for a topic.
        """

        return {
            "topic_id": topic_id,
            "keywords": self.topic_model.get_topic(topic_id),
            "label": self.topic_labels.get(topic_id, None)
        }
    
    def find_nearest_topics(self, query: str, top_n: int = 5):
        """
        Find closest topics to a query string.
        """

        return self.topic_model.find_topics(query, top_n=top_n)

    def save(self, path: str):
        self.topic_model.save(path)
        
        
if __name__ == "__main__":
    df =  pd.read_parquet("../Data/Clean/Dataset_with_clusters.parquet")
    model = SemanticTopicModel()

    model.fit(
        texts=df["Content"].tolist(),
        embeddings=np.vstack(df["Embedding"].values),
        clusters=df["Cluster"].tolist()
    )

    model.label_clusters()
    # model.save("../Models/topic_model.joblib")
    
    ## At inference
    model = SemanticTopicModel(model_path="../Models/topic_model.joblib")

    topics, probs = model.transform([
        "The prime minister announced new election reforms."
    ])
    
    df = model.add_semantic_topics(df, "Content")

