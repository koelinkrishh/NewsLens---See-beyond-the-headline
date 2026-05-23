import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# NLP Heuristics libs
import spacy
import nltk
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# LangChain & Pydantic imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Core system imports
from src.config import *
from src.logger import logging
from src.exception import CustomException
from src.Components.utils import clean_text

# Load environment variables cleanly
load_dotenv()

"""
Use LLM for deeper Feature extraction -> Only to be used on inference article, not entire dataset.
"""


class SemanticFeatures(BaseModel):
    tone: str = Field(description="The dominant tone of the article (e.g., Analytical, Sensationalist, Critical, Objective, Alarmist)")
    objectivity_score: float = Field(description="A score between 0.0 (highly biased/opinionated) and 1.0 (highly objective/factual)")
    primary_category: str = Field(description="The primary news category (e.g., Politics, Business, Technology, Science, Health, Sports, Entertainment)")

class TextFeatureGenerator:
    """
    Modernized hybrid Feature Generator.
    Combines classical text statistics and POS tag counts (using local CPU libraries)
    with high-level semantic metadata (using a LangChain structured LLM extraction chain).
    """

    def __init__(self, 
        spacy_model: str = "en_core_web_sm", 
        provider: str = "huggingface", 
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
        api_key: Optional[str] = None,
        semantic:bool = True
    ):
        logging.info(f"⚡ Initializing Hybrid Feature Engine [{spacy_model}] [Provider: {provider}, Model: {model}]...")
        
        try:
            self.semantic = semantic
            
            # 1. Download NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            
            # 2. Load local SpaCy model for classical POS/NER statistics
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                logging.warning(f"Model `{spacy_model}` not found. Downloading...")
                from spacy.cli import download
                download(spacy_model)
                self.nlp = spacy.load(spacy_model)
            
            if self.semantic:
                # 3. Initialize LangChain LLM for structured semantic extraction
                self.provider = provider.lower()
                self.model = model
                
                if self.provider == "groq":
                    key = api_key or os.getenv("GROQ_API_KEY")
                    if not key:
                        raise ValueError("GROQ_API_KEY is missing. You must provide a valid Groq API Key.")
                    self.llm = ChatGroq(model=self.model, groq_api_key=key, temperature=0)
                    
                elif self.provider == "openai":
                    key = api_key or os.getenv("OPENAI_API_KEY")
                    if not key:
                        raise ValueError("OPENAI_API_KEY is missing. You must provide a valid OpenAI API Key.")
                    self.llm = ChatOpenAI(model=self.model, openai_api_key=key, temperature=0)
                    
                elif self.provider == "huggingface":
                    key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    if not key:
                        raise ValueError("HF_TOKEN / HUGGINGFACEHUB_API_TOKEN is missing. You must provide a valid Hugging Face Token.")
                    self.llm = ChatOpenAI(
                        model=self.model,
                        openai_api_key=key,
                        base_url="https://router.huggingface.co/v1",
                        temperature=0
                    )
                    
                elif self.provider == "gemini":
                    key = api_key or os.getenv("GOOGLE_API_KEY")
                    if not key:
                        raise ValueError("GOOGLE_API_KEY is missing. You must provide a valid Gemini API Key.")
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self.llm = ChatGoogleGenerativeAI(model=self.model, google_api_key=key, temperature=0)
                else:
                    raise ValueError(f"Unsupported provider: '{self.provider}'. Choose from 'groq', 'openai', 'huggingface', or 'gemini'.")
            
            logging.info("Hybrid NLP resources and LLM initialized successfully.")
            
        except Exception as e:
            logging.error(f"Failed to initialize TextFeatureGenerator: {str(e)}")
            raise CustomException(e, sys)
        
    def extract_classical_features(self, text: str, doc) -> dict:
        """
        Extracts structural, vocabulary, and syntactic counts using fast local libraries.
        """
        features = {}
        sentences = sent_tokenize(text)
        tokens = [t.lower() for t in word_tokenize(text) if t.isalnum()]
        counts = Counter(tokens)
        stopwords_used = [t for t in tokens if t in self.stop_words]
        
        rare_words = [w for w, c in counts.items() if c == 1]
        pos_counts = Counter(tok.pos_ for tok in doc if tok.is_alpha)
        ent_counts = Counter(ent.label_ for ent in doc.ents)
        
        # 1. Text length and Structure
        features["char_count"] = len(text)
        features["char_count_no_spaces"] = len(text.replace(" ", ""))
        features["sentence_count"] = len(sentences)
        
        # 2. Vocabulary and Lexical richness
        features["word_count"] = len(tokens)
        features["unique_word_count"] = len(counts)
        features["lexical_diversity"] = features["unique_word_count"] / max(1, features["word_count"])
        features["hapax_ratio"] = len(rare_words) / max(1, features["word_count"])
        
        # 3. Stop-words and content words
        features["stopword_Count"] = len(stopwords_used)
        features["content_word_count"] = len(tokens) - len(stopwords_used)
        features["stopword_ratio"] = len(stopwords_used) / max(1, features["word_count"])
        
        # 4. POS tag counting
        features["noun_count"] = pos_counts.get("NOUN", 0)
        features["verb_count"] = pos_counts.get("VERB", 0)
        features["adj_count"] = pos_counts.get("ADJ", 0)
        features["adv_count"] = pos_counts.get("ADV", 0)
        features["pronoun_count"] = pos_counts.get("PRON", 0)
        
        # 5. Entity tag counting
        features["person_count"] = ent_counts.get("PERSON", 0)
        features["org_count"] = ent_counts.get("ORG", 0)
        features["gpe_count"] = ent_counts.get("GPE", 0)
        features["event_count"] = ent_counts.get("EVENT", 0)
        features["unique_entity_count"] = len(set(ent.text for ent in doc.ents))
        
        # 6. Readability metrics
        features["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
        features["gunning_fog"] = textstat.gunning_fog(text)
        
        return features

    def extract_semantic_features(self, text: str) -> dict:
        """
        Invokes LangChain structured outputs to extract high-level semantic metadata.
        Uses clean fallback logic if API is offline or missing.
        """
        if not self.llm:
            return {
                "semantic_tone": "Unknown",
                "semantic_objectivity_score": 0.5,
                "semantic_primary_category": "General"
            }
            
        try:
            # Analyze a clean snippet (up to 3000 chars) to keep tokens low and processing fast
            truncated_text = text
            structured_llm = self.llm.with_structured_output(SemanticFeatures)
            
            prompt = f"Analyze the following news article text and extract key semantic features:\n\n{truncated_text}"
            result = structured_llm.invoke(prompt)
            
            return {
                "semantic_tone": result.tone,
                "semantic_objectivity_score": result.objectivity_score,
                "semantic_primary_category": result.primary_category,
            }
        except Exception as e:
            logging.warning(f"Failed to extract semantic features via LLM: {str(e)}. Using fallback defaults.")
            return {
                "semantic_tone": "Unknown",
                "semantic_objectivity_score": 0.5,
                "semantic_primary_category": "General"
            }

    def extract_features(self, text: str) -> dict:
        """
        Extracts both classical heuristics and LangChain-based semantic features.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return {}
                
            cleaned = clean_text(text)
            doc = self.nlp(cleaned)
            
            # Get local counts and readability scores
            classical = self.extract_classical_features(cleaned, doc)
            
            # Get LLM-based metadata
            if self.semantic:
                semantic = self.extract_semantic_features(cleaned)
            else:
                semantic = {}
            
            # Combine both dictionaries
            return {**classical, **semantic}
            
        except Exception as e:
            logging.error(f"Error extracting hybrid features: {str(e)}")
            raise CustomException(e, sys)
        
    def process_dataframe(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Applies cleaning and hybrid feature extraction to the dataframe.
        """
        try:
            logging.info(f"Processing DataFrame. Shape: {df.shape}")
            
            logging.info(f"Cleaning column: {text_col}")
            df[text_col] = df[text_col].astype(str).apply(clean_text)
            
            logging.info("Extracting hybrid features (Classical & LLM-based)...")
            features_list = df[text_col].apply(self.extract_features).tolist()
            features_df = pd.DataFrame(features_list, index=df.index)
            
            df_final = pd.concat([df, features_df], axis=1)
            logging.info(f"Feature generation complete. New shape: {df_final.shape}")
            
            return df_final
        except Exception as e:
            raise CustomException(e, sys)
        
    def Create_features(self, article: str | pd.DataFrame, text_col: str = None) -> pd.DataFrame:
        """
        Unified inference and batch generation interface.
        """
        try:
            if isinstance(article, str):
                features = self.extract_features(article)
                return pd.DataFrame(features, index=[0])
            elif isinstance(article, pd.DataFrame):
                if text_col is None:
                    raise ValueError("text_col is required when passing a dataframe")
                return self.process_dataframe(article, text_col)
            else:
                raise CustomException(f"Unsupported input type: {type(article)}", sys)
        except Exception as e:
            raise CustomException(e, sys)

# --- Local Verification script ---
if __name__ == "__main__":
    print("Testing Clean LangChain FeatureGenerator...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        processor = TextFeatureGenerator(provider="huggingface", model="meta-llama/Meta-Llama-3-8B-Instruct", semantic=False)
        test_text = """NEW DELHI: The Indian space agency ISRO on Tuesday successfully launched its next-generation meteorological satellite on board a GSLV rocket from Sriharikota,
        aimed at significantly boosting weather forecasting, storm tracking and disaster warnings across the subcontinent.
        The launch went flawlessly under clear blue skies."""
        
        print("Running single article hybrid extraction...")
        feats_df = processor.Create_features(test_text)
        
        print("\nGenerated Features:\n")
        for col in feats_df.columns:
            print(f"- {col}: {feats_df.iloc[0][col]}")
        
        print("\n[SUCCESS] HYBRID FEATURE GENERATION VERIFIED SUCCESSFUL!")
        
    except Exception as e:
        print(f"Test run encountered an issue: {e}")
