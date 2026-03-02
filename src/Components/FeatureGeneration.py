import re
import html
import unicodedata
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from typing import Dict, Any, List, Optional
import sys, os

    # NLP libs
import spacy
import nltk
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from .utils import clean_text

# Logging
from src.config import * # Loading config paths
from src.logger import logging
from src.exception import CustomException

# 1. Suppress the oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Suppress other TensorFlow logging (0=all, 1=no INFO, 2=no INFO/WARN, 3=no ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TextFeatureGenerator:
    """
    Unified feature generation class for news article analysis.
    Each sub-task uses the most appropriate NLP library.
    """

    def __init__(self, spacy_model:str="en_core_web_sm", enable_semantic:bool=True):
        logging.info(f"⚡ Initializing Heuristics Engine [{spacy_model}]...")
        
        try:
            logging.info(f"Loading spaCy model: {spacy_model}")
            
            # 1. Download NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                logging.warning(f"Model `{spacy_model}` not found. Downloading...")
                from spacy.cli import download
                download(spacy_model)
                self.nlp = spacy.load(spacy_model)
                
            logging.info("NLP resources initialized successfully.")
                
        except Exception as e:
            raise CustomException(e, sys)
        
    # Generate features ->
    def extract_features(self, text:str) -> dict:
        """
        Extracts features from the given text.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return {}
            text = clean_text(text)
            
            doc = self.nlp(text)
            
            features = {}
            sentences = sent_tokenize(text)
            tokens = [t.lower() for t in word_tokenize(text) if t.isalnum()]
            counts = Counter(tokens)
            stopwords_used = [t for t in tokens if t in self.stop_words]
            
            rare_words = [w for w,c in counts.items() if c==1]
            
            pos_counts = Counter(tok.pos_ for tok in doc if tok.is_alpha)
            ent_counts = Counter(ent.label_ for ent in doc.ents)
            
            # 1. Text length and Structure
            features["char_count"] = len(text)
            features["char_count_no_spaces"] = len(text.replace(" ",""))
            features["sentence_count"] = len(sentences)
            
            # 2. Vocabulary and Lexical richness
            features["word_count"] = len(tokens)
            features["unique_word_count"] = len(counts)
            features["lexical_diversity"] = features["unique_word_count"]/max(1,features["word_count"])
            features["hapax_ratio"] = len(rare_words)/max(1,features["word_count"])
            
            # 3. Stop-words and content words
            features["stopword_Count"] = len(stopwords_used)
            features["content_word_count"] = len(tokens) - len(stopwords_used)
            features["stopword_ratio"] = len(stopwords_used)/max(1,features["word_count"])
            
            # 4. POS
            features["noun_count"] = pos_counts.get("NOUN", 0)
            features["verb_count"] = pos_counts.get("VERB", 0)
            features["adj_count"] = pos_counts.get("ADJ", 0)
            features["adv_count"] = pos_counts.get("ADV", 0)
            features["pronoun_count"] = pos_counts.get("PRON", 0)
            
            # 5. NER
            features["person_count"] = ent_counts.get("PERSON", 0)
            features["org_count"] = ent_counts.get("ORG", 0)
            features["gpe_count"] = ent_counts.get("GPE", 0)
            features["event_count"] = ent_counts.get("EVENT", 0)
            features["unique_entity_count"] = len(set(ent.text for ent in doc.ents))
            
            # 6. Readibility features
            features["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
            features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            features["gunning_fog"] = textstat.gunning_fog(text)
            
            return features
        
        except Exception as e:
            logging.error(f"Error extracting features for text: {text[:30]}...")
            raise CustomException(e, sys)
        
    def process_dataframe(self, df:pd.DataFrame, text_col:str) -> pd.DataFrame:
        """
        Applies cleaning and feature extraction to the dataframe.
        """
        try:
            logging.info(f"Processing DataFrame. Shape: {df.shape}")
            
            logging.info(f"Cleaning column: {text_col}")
            df[text_col] = df[text_col].apply(clean_text)
            logging.info("Extracting features (this may take time)...")
            
            features_list = df[text_col].apply(self.extract_features).tolist()
            features_df = pd.DataFrame(features_list, index=df.index)
            
            # concatenation
            df_final = pd.concat([df, features_df], axis=1)
            logging.info(f"Feature generation complete. New shape: {df_final.shape}")
            
            return df_final
        except Exception as e:
            raise CustomException(e, sys)
        
    ## Inference call
    def Create_features(self, article:str|pd.DataFrame, text_col:str=None) -> pd.DataFrame:
        try:
            if isinstance(article, str): # Given only text
                features = self.extract_features(article)
                logging.info("Successfully created features.")
                return pd.DataFrame(features, index=[0])
            elif isinstance(article, pd.DataFrame): # Given a dataframe
                if text_col is None:
                    logging.error("Textual column name missing for Inference call")
                    raise ValueError("text_col is required when article is a dataframe")
            
                return self.process_dataframe(article, text_col)
            else:
                raise CustomException(f"Unsupported type: {type(article)}", sys)
        except Exception as e:
            raise CustomException(e, sys)
            

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Configuration (You can change paths here)
        INPUT_PATH = DATA_SAMPLE_CSV
        OUTPUT_PATH = CLEANED_DATASET_PARQUET
        TEXT_COLUMN = "Content"

        logging.info("Starting Data Cleaning & Feature Generation Pipeline")
        
        processor = TextFeatureGenerator()
        
        # 1. Load Data
        if os.path.exists(INPUT_PATH):
            logging.info(f"Loading data from {INPUT_PATH}")
            if INPUT_PATH.endswith('.csv'):
                df = pd.read_csv(INPUT_PATH, index_col=0)
            else:
                df = pd.read_parquet(INPUT_PATH)
        else:
            raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
            
        # 2. Process
        df_processed = processor.process_dataframe(df, text_col=TEXT_COLUMN)
        print(df_processed.head())
        # 3. Save
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        # df_processed.to_parquet(OUTPUT_PATH)
        logging.info(f"Successfully saved processed data to {OUTPUT_PATH}")
        
    except Exception as e:
        logging.error("Pipeline Failed")
        print(e)
    
    