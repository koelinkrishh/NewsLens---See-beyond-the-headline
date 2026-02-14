import faiss
import logging

# Configure logging for production-grade output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SearchEngine")

from sklearn.metrics.pairwise import cosine_similarity

def advance_search(query:str|List[str], model:BERTopic, df:pd.DataFrame,
    embeddings:np.ndarray=None, top_k:int=5, filter_space:bool=True):
    """
    Advance semantic search function with Topic Routing
    
    Args:
        query (str | List[str]): Query Article or list of queries
        model (BERTopic): Fitted BERTopic model pipeline
        df (pd.DataFrame): Dataset
        embeddings (np.ndarray): Topic embeddings from BERTopic
        top_k (int, optional): No. of results to return.
        filter_space (bool, optional): Filter search space based on Topic Cluster matching.

    Returns:
        pd.DataFrame: Top K similar articles
    """
    if embeddings is None:
        embeddings = np.vstack(df['Embedding'])
    
    # 1. Embed the query
    query_vec = model.embedding_model.embed_documents([query])
    # 2. Predict topic of query (Zero-Shot Routing)
    pred_topic, prob = model.transform([query])
    
    # 3. Filter search space (Optimization)
    search = df.index
    if filter_space and pred_topic[0] != -1:
        mask = (df['Topic'] == pred_topic[0])
        if mask.sum() > 0:
            search = df[mask].index
        
    if search.shape[0] == 0:
        search = df.index # Falling back to entire dataset
    
    # Filter embeddings
    filter_em = embeddings[search]
    
    # 4. Semantic search
    sim_search = cosine_similarity(query_vec, filter_em)
    # 5. Get Top K
    best_local_idx = np.argsort(sim_search)[0][::-1][:top_k]
    # print(np.argsort(sim_search), best_local_idx)
    
    # similar_embeddings =  filter_em[best_local_idx]
    return df.loc[best_local_idx]


class SearchEngine:
    def __init__(self, model:BERTopic, df:pd.DataFrame, embeddings:np.ndarray=None):
        """
        Initializes a FAISS Search Engine
        creates seperate vector index for each Topic to maximize speed and relevence
        """
        if embeddings is None:
            embeddings = np.vstack(df['Embedding'].values)
        print(embeddings.shape)
        
        self.topic_model = model
        self.df = df
        
        self.indices = {}   # Dict to store FAISS index per topic
        self.id_maps = {}   # Dict to store mapping from FAISS ID -> Dataframe index
        
        # 1. Normalize Embeddings -> Inner Product == Cosine Similarity
        self.embeddings = self._normalize(embeddings)
        self.em_dim = self.embeddings.shape[1]
        
        self._build_indices(df)
        
    def _normalize(self, v):
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return (v/norm).astype("float32")
        
    def _build_indices(self, df:pd.DataFrame):
        """
        Build a separate IndexFlatIP for each topic cluster
        """
        unique_topics = df['Topic'].unique()
        
        for topic_id in unique_topics:
            # 1. Boolean mask for topic match
            mask = (df['Topic']==topic_id).values
            # 2. Specific embedding for this cluster
            subset_embedding = self.embeddings[mask]
            # 3. Store the original DataFrame indices to map back
            self.id_maps[topic_id] = df.index[mask].values
            
            # 4. Create FAISS index (Inner product)
            idx = faiss.IndexFlatIP(self.em_dim)    # Inner Product
            """ For larger size datasets, switch to IndexIVFFlat | IndexHNSW """
            idx.add(subset_embedding) # add training set
            
            self.indices[topic_id] = idx
        
        # Adding complete dataset as well
        self.id_maps[-1] = df.index
        idx = faiss.IndexFlatIP(self.em_dim)
        idx.add(self.embeddings)
        self.indices[-1] = idx
            
    def search(self, query:str, top_k:int = 5) -> pd.DataFrame:
        """
        Perform a routed semantic search
        
        return: 
            similar articles info, distance based score, cluster id
        """
        # 1. Embed Query
        query_vec = self.topic_model.embedding_model.embed([query])
        query_vec = self._normalize(query_vec).astype("float32")
        
        # 2. Topic Routing (zero-shot)
        pred_topics, _ = self.topic_model.transform(query)
        target_topic = pred_topics[0]
        
        # 3. Fallback
        if target_topic not in self.indices:
            target_topic = -1 # Complete dataset
            
        # 4. Sharded search
        index = self.indices[target_topic]
        k = min(top_k, index.ntotal) # safety, dont request more then cluster size
        
        # D = Distances(Scores), I=Indices (FAISS)
        D, I = index.search(query_vec, k)
        # print(D[0], I[0])
        
        # 5. Map FAISS ids back to get dataframe
        df_index = self.id_maps[target_topic][I[0]]
        scores = D[0]
        
        return self.df.loc[df_index], scores, target_topic
        