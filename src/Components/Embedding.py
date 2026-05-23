import sys
import os
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Tuple

# --- Local Infrastructure Imports ---
from src.config import *
from src.logger import logging
from src.exception import CustomException
# Loading utility functions (assuming clean_text is here)
from src.Components.utils import clean_text

# Chaching for faster processing
from functools import lru_cache

# 1. Suppress the oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ArticleEmbeddingEngine:
    """
    High-performance embedding engine.
    Optimized for speed using batch processing and vectorization.
    """
    def __init__(self, model_name:str = SENTENCE_TRANSFORMER_MODEL, 
            spacy_model:str = SPACY_MODEL,
            max_tokens:int = 512, 
            batch_size:int = 32, 
            overlap_sentences:int = 1):
        
        logging.info(f"Initializing Embedding Engine [{model_name}]...")
        try:
            self.max_tokens = max_tokens
            self.batch_size = batch_size
            self.overlap_sentences = overlap_sentences
            
            # 1. Load SBERT Model
            self.model = SentenceTransformer(model_name)
            self.model.max_seq_length = max_tokens
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # Access tokenizer directly for length checks
            self.tokenizer = self.model.tokenizer
            
            logging.info(f"Model loaded. Dimension: {self.embedding_dim}, Max Tokens: {max_tokens}")
            
            # 2. Load Spacy (Sentencizer only for speed)
            self.nlp = spacy.load(spacy_model, disable=["tagger", "ner", "lemmatizer"])
            self.nlp.add_pipe("sentencizer")
            
        except Exception as e:
            logging.error("Error during embedding engine initialization.")
            raise CustomException(e, sys)

    def _smart_chunking(self, sentences: List[str]) -> Tuple[List[str], List[int]]:
        """
        Optimized chunking logic.
        1. Batch-calculates lengths of all sentences first (Fast).
        2. Groups sentences into chunks using integer arithmetic.
        """
        if not sentences:
            return [], []
        
        # SPEED OPTIMIZATION: Batch tokenize to get lengths in one go
        # This replaces the slow loop: [len(tokenizer(s)) for s in sentences]
        tokenized_inputs = self.tokenizer(sentences, add_special_tokens=False, padding=False, truncation=False)
        sent_lengths = [len(ids) for ids in tokenized_inputs['input_ids']]
        
        chunks = []
        chunk_weights = [] # store token count for weighted pooling later
        
        curr_chunk_sents = []
        curr_length = 0
        
        for i, length in enumerate(sent_lengths):
            # Check for single sentences that are too long
            if length > self.max_tokens:
                logging.warning(f"⚠️ Found sentence longer than model limit ({length} > {self.max_tokens}). It will be truncated.")
            
            # Flush if adding this sentence exceeds limit
            if curr_length + length > self.max_tokens:
                # Add current chunk
                if curr_chunk_sents:
                    chunks.append(" ".join(curr_chunk_sents))
                    chunk_weights.append(curr_length)
                
                # Apply Overlap
                if self.overlap_sentences > 0 and len(curr_chunk_sents) > self.overlap_sentences:
                    overlap_sents = curr_chunk_sents[-self.overlap_sentences:]
                    # Re-calculate length of overlap (approximate or look up)
                    overlap_len = sum(sent_lengths[i-len(overlap_sents):i])
                    
                    curr_chunk_sents = overlap_sents + [sentences[i]]
                    curr_length = overlap_len + length
                else:
                    curr_chunk_sents = [sentences[i]]
                    curr_length = length
            else:
                curr_chunk_sents.append(sentences[i])
                curr_length += length
                
        # Add last chunk
        if curr_chunk_sents:
            chunks.append(" ".join(curr_chunk_sents))
            chunk_weights.append(curr_length)
            
        return chunks, chunk_weights
    
    
    def Embed_dataframe(self, df: pd.DataFrame, text_col: str):
        """
        Main Pipeline:
        1. Clean Text
        2. Split Sentences (Batch)
        3. Chunk (Smart)
        4. Embed (Batch)
        5. Pool (Weighted)
        """
        logging.info("Starting Batch Processing...")
        
        # 1. Clean Texts
        texts = df[text_col].astype(str).apply(clean_text).tolist()
        
        all_chunks = []
        doc_map = []  # Stores (start_index, end_index) in all_chunks for each doc
        all_weights = [] # Token counts for weighting
        
        # 2. Batch Sentence Splitting (using nlp.pipe for speed)
        # This is much faster than looping df.apply(nlp)
        logging.info("Splitting sentences...")
        doc_stream = self.nlp.pipe(texts, batch_size=64, n_process=1)
        
        for doc in tqdm(doc_stream, total=len(texts), desc="Chunking Docs"):
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            chunks, weights = self._smart_chunking(sentences)
            
            if not chunks:
                doc_map.append(None) # Mark empty docs
                continue
                
            start_idx = len(all_chunks)
            all_chunks.extend(chunks)
            all_weights.extend(weights)
            end_idx = len(all_chunks)
            
            doc_map.append((start_idx, end_idx))
            
        # 3. Embed All Chunks
        logging.info(f"Embedding {len(all_chunks)} chunks...")
        if len(all_chunks) > 0:
            chunk_embeddings = self.model.encode(
                all_chunks,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True # Crucial for cosine similarity later
            )
        else:
            chunk_embeddings = np.array([])

        # 4. Re-assemble (Weighted Pooling)
        final_embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        logging.info("Pooling chunks into document embeddings...")
        for doc_idx, indices in enumerate(doc_map):
            if indices is None:
                continue # Stays zero vector
            
            start, end = indices
            
            # Get embeddings for this doc's chunks
            embs = chunk_embeddings[start:end]
            
            # Get weights (lengths)
            weights = np.array(all_weights[start:end], dtype=np.float32)
            
            # Safety: avoid divide by zero
            if weights.sum() == 0:
                weights = np.ones_like(weights)
                
            # Weighted Average
            weights = weights / weights.sum()
            pooled = np.average(embs, axis=0, weights=weights)
            
            # Normalize final vector
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
                
            final_embeddings[doc_idx] = pooled

        return final_embeddings
    
    @lru_cache(maxsize=64) # store the last 64 article embedding
    def create_embedding(self, article: str):
        """
        Inference pipeline for a single article.
        Returns a normalized embedding vector of shape (embedding_dim,)
        """
        if article is None or not str(article).strip():
            logging.warning("Empty article was given to create embedding.")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        logging.info("Creating Embedding for Inference article.")
        
        text = clean_text(article)
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        
        chunks, weights = self._smart_chunking(sents)
        
        if not chunks:
            logging.warning("Problem with chunking text!")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        inference_embed = self.model.encode(
            chunks, batch_size=self.batch_size, convert_to_numpy=True,
            show_progress_bar=True, normalize_embeddings=True # Crucial for cosine similarity later
        )
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        
        pooled = np.average(inference_embed, axis=0, weights=weights).astype(np.float32)
        norm = np.linalg.norm(pooled)
        if norm>0: pooled = pooled / norm
        
        return pooled
    
    # For compatibility with BERTopic module
    @lru_cache(maxsize=64)
    def embed(self, texts:List[str]): 
        """
        Compatibility method for BERTopic.
        Allows BERTopic to use this engine (and its cache) directly.
        """
        return self.Embed_dataframe(pd.DataFrame({TEXT_COL:texts}), TEXT_COL)

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        # Configuration
        INPUT_FILE = CLEANED_DATASET_PARQUET # from config.py
        OUTPUT_FILE = EMBEDDED_DATASET       # from config.py
        TEXT_COL = "Content"
        
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
        
        logging.info(f"Loading data from {INPUT_FILE}")
        df = pd.read_parquet(INPUT_FILE)
        
        # Init Engine
        engine = ArticleEmbeddingEngine(batch_size=64) # Increased batch size for speed
        
        # Run
        embeddings = engine.Embed_dataframe(df, TEXT_COL)
        
        # Save
        df["embedding"] = list(embeddings) # Convert to list for parquet compatibility
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        # df.to_parquet(OUTPUT_FILE, index=False)
        
        logging.info(f"✅ Success! Saved to {OUTPUT_FILE}")
        
    except Exception as e:
        logging.error(f"Pipeline Failed: {e}")
        raise CustomException(e, sys)

