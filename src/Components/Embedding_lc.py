import os
import sys
import numpy as np
from typing import List

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

# Core system imports
from src.logger import logging
from src.exception import CustomException
from src.Components.utils import clean_text

# Load environment variables cleanly
from dotenv import load_dotenv
load_dotenv()

class ArticleEmbeddingEngine:
    """
    LangChain-based Embedding Engine that splits articles and encodes chunks
    using standard OpenAI or HuggingFace models at their native dimensions.
    """
    def __init__(self, 
        embedding_type: str = "huggingface_local", 
        chunk_size: int = 1000, 
        chunk_overlap: int = 150
        ):
        
        logging.info(f"Initializing ArticleEmbeddingEngine (Type: {embedding_type})")
        
        try:
            self.embedding_type = embedding_type.lower()
            
            # Standard LangChain Recursive Chunker
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Defer heavy model instantiation until first use to avoid network calls
            # Store config and only create the encoder lazily when an embed method is called.
            self._encoder_config = {
                "type": self.embedding_type,
            }
            # Optional: allow model name override from params.yaml or env
            # The actual encoder object will be set by _ensure_encoder()
            self.encoder = None
            
            # Map known dimensions immediately to support lazy initialization without eager probing
            if self.embedding_type == "openai":
                self.dimension = 3072
            elif self.embedding_type == "gemini":
                self.dimension = 768
            else:
                self.dimension = 384
                
            logging.info(f"ArticleEmbeddingEngine configured for lazy initialization (Type: {self.embedding_type}, Dimension: {self.dimension}).")
            
        except Exception as e:
            logging.error(f"Failed to initialize Embedding Engine: {str(e)}")
            raise CustomException(e, sys)

    def chunk_article(self, text: str) -> List[str]:
        """
        Cleans and splits a raw text document using the RecursiveCharacterTextSplitter.
        """
        if not text or not str(text).strip():
            return []
        cleaned = clean_text(text)
        return self.splitter.split_text(cleaned)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generates embeddings for a batch of text chunks, L2 normalized.
        Returns a 2D numpy array of shape (num_chunks, native_dimension).
        """
        try:
            if not chunks:
                return np.empty((0, self.dimension), dtype=np.float32)
                
            # Ensure encoder is instantiated lazily
            self._ensure_encoder()
            raw_embeddings = self.encoder.embed_documents(chunks)
            
            normalized = []
            for emb in raw_embeddings:
                arr = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                normalized.append(arr)
                
            return np.vstack(normalized)
            
        except Exception as e:
            logging.error(f"Error during chunk embedding: {str(e)}")
            raise CustomException(e, sys)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates an L2 normalized 1D embedding vector for a single search query.
        Returns a vector of shape (native_dimension,).
        """
        try:
            cleaned = clean_text(query)
            self._ensure_encoder()
            raw_emb = self.encoder.embed_query(cleaned)
            
            arr = np.array(raw_emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr
            
        except Exception as e:
            logging.error(f"Error embedding query: {str(e)}")
            raise CustomException(e, sys)

    def _ensure_encoder(self):
        """Instantiate the chosen encoder on first use and discover its dimension."""
        if self.encoder is not None:
            return
        try:
            etype = self._encoder_config.get("type")
            if etype == "openai":
                self.encoder = OpenAIEmbeddings(model="text-embedding-3-large")
            elif etype == "huggingface_api":
                self.encoder = HuggingFaceEndpointEmbeddings(model="BAAI/bge-small-en-v1.5")
            elif etype == "huggingface_local":
                self.encoder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            elif etype == "gemini":
                self.encoder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
            else:
                raise ValueError(f"Unsupported embedding type: '{etype}'")

            # Probe dimension safely
            try:
                test_embed = self.encoder.embed_query("test")
                self.dimension = len(test_embed)
            except Exception:
                # If probe fails, leave dimension None; callers should handle accordingly
                logging.warning("Unable to probe encoder dimension during initialization; will infer on first embed.")
                self.dimension = None
            logging.info(f"Encoder instantiated lazily (type={etype}). Dimension={self.dimension}")
        except Exception as e:
            logging.error(f"Failed to instantiate encoder: {e}")
            raise CustomException(e, sys)

# --- Local Verification script ---
if __name__ == "__main__":
    print("Testing Clean LangChain Embedding Engine...")
    try:
        # Standard lightweight local run
        engine = ArticleEmbeddingEngine(embedding_type="huggingface_api")
        test_text = """NEW DELHI: Amid continuing scrutiny over the NEET-UG 2026 paper leak controversy, the National Testing Agency (NTA) on Tuesday announced a series of institutional, technological and administrative reforms while union education minister Dharmendra Pradhan chaired a high-level review meeting to assess preparations for the forthcoming NEET (UG) re-examination and tighten security arrangements across states.
            The reforms come weeks after the cancellation of the May 3 NEET-UG examination following allegations that parts of the question paper had been leaked before the test. The controversy revived concerns over examination security after earlier irregularities linked to NEET in 2024, including paper leak allegations, grace marks disputes, suspicious perfect scores and claims of organised malpractice networks operating across multiple states.
            NTA said it had initiated measures to strengthen its “leadership team, institutional capacity and oversight mechanisms” in line with recommendations of the high-level expert committee headed by former ISRO chief K Radh  akrishnan.
            As part of the restructuring, four senior officers, including two joint secretary-level officers designated as additional director generals, have been posted to the agency to improve administrative oversight and o   perational monitoring.
            The NTA has also advertised three specialist leadership posts — chief technology officer (CTO), chief finance officer (CFO) and general manager (human resources) — aimed at modernising examination systems, institutional governance and i    nternal accountability structures.
            According to the agency, the proposed CTO will oversee the full digital examination ecosystem, including confidential question paper management, AI-driven integrity controls, biometric and facial authentication systems, cyber-security safeguards and anomaly analytics designed to d   etect suspicious patterns during examinations.
            The agency said broader reforms would include structural changes in question paper preparation, translation, printing and logistics, along with technology-enabled safeguards at every stage. It also proposed continuous monitoring systems, stronger audit frameworks, professional training, enhanced stakeholder coordination and improved grievance redressal mechanisms for students and parents.
            During Tuesday’s review meeting, Pradhan stressed that all gaps identified in the earlier examination process must be comprehensively addressed and eliminated. He directed officials to ensure the re-examination was conducted in a “secure, seamless and foolproof manner” under stringent protocols. The minister also instructed authorities to hold coordination meetings with district magistrates and superintendents of police across states to strengthen monitoring, maintain vigilance at centres and ensure adequate arrangements, transportation support, medical assistance and student facilitation systems for candidates appearing in the examination nationwide over the coming weeks. Officials were also asked to ensure uninterrupted power supply, secure storage facilities for examination materials and rapid-response teams for emergencies.
            """
        
        chunks = engine.chunk_article(test_text)
        print(f"Chunks generated:")
        for i,ck in enumerate(chunks):
            print(f"Chunk {i+1}:\n{ck}")
            print("*"*10)
        
        chunk_vecs = engine.embed_chunks(chunks)
        print(f"Embeddings shape: {chunk_vecs.shape} (Expected: (num_chunks, vector_dimension))")
        if len(chunk_vecs) > 0:
            print(f"L2 Norm of first chunk: {np.linalg.norm(chunk_vecs[0]):.4f}")
        print("[SUCCESS] STUB VERIFIED SUCCESSFUL!")
        
    except Exception as ex:
        print(f"Test run encountered an issue: {ex}")
        
