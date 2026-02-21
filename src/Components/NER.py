"""
NER - predefined class + NER - Blind class match + Keyword Extraction

This Module is a Complete Inference Pipeline.
"""
import numpy as np
import pandas as pd
import os, sys, torch, spacy
from typing import List, Tuple, Dict, Optional

# --- Model Imports ---
from gliner2 import GLiNER2
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# --- Local Imports ---
from src.config import *
from src.logger import logging
from src.exception import CustomException


class InformationExtractor:
    """
    High-Fidelity Information Extraciton Engine
    Combines spacy_trf for predefined class, GLiner for Zero-shot NER and KeyBERT for semantic Keyword Extraction.
    """
    
    default_ner_labels = [
        "Person", "Celebrity", "Political party", "Politician", "Activist", "Criminal", "Victim", "Witness",
        "Profession", "Job title", "Author", "Scientist", "Journalist", "Speaker", "Writer", "Artist",
        "Affiliation", "Organisation", "Company", "Startup", "Institution", "College", "University", "Government agency", "Military organisation", "Union", "Sports team", "Media",
        "Country", "City", "State", "Region", "Continent", "Climate zone", "Forest", "Desert", "Mountain", "Park", "Water body",
        "Building", "Airport", "Monument", "Landmark", "Date", "Time", "Duration", "Percent", "Money", "Temperature", "Speed", "Age",
        "Law", "Case", "Judge", "Constitution", "Election", 
        "Medical", "Disease", "Drug", "Symptom", "Chemical"
        "Location", "Product", "Event", "Work_of_art", "Language",
        "Business", "Market", "Stock", "Currency", "Product"
        "Sport", "Games", "Award", "Event",
        "Art", "Book", "Movie", "TV show", 
        "Computer", "Vehicle", "Machine", "Programming language", "Technology",
        "Color", "Shape", "Size", "Weight", "Weapon", "Battle", "Natural disaster",
        "Quantity", "Ordinal", "Cardinal", 
        "Animal", "Plant", "Organism",
    ]
    
    def __init__(self, spacy_model_name:str = Spacy_trf, gliner_model_name:str = Gliner_ner, keybert_model_name:str = Keybert_model,
        sentence_transformer_model:str = SENTENCE_TRANSFORMER_MODEL
    ):
        logging.info("Initializing Information Extractor...")
        try:
            # 1. Load spaCy (Predefined 18-class NER)
            logging.info(f"Loading spaCy NER model: {spacy_model_name}")
            try:
                self.nlp = spacy.load(spacy_model_name)
            except OSError:
                from spacy.cli import download
                download(spacy_model_name)
                self.nlp = spacy.load(spacy_model_name)
                
            # 2. Load GLiNER (Zero-Shot Custom NER)
            logging.info(f"Loading GLiNER model: {gliner_model_name}")
            self.gliner_model = GLiNER2.from_pretrained(gliner_model_name)
            
            # 3. Load KeyBERT (Semantic Keywords)
            logging.info(f"Loading KeyBERT model: {keybert_model_name}")
            self.kw_model = KeyBERT(model=keybert_model_name)
            
            # 4. Load Sentence Transformer
            self.sentence_transformer_model = SentenceTransformer(sentence_transformer_model)
            logging.info("Information Extractor initialized successfully.")
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def extract_spacy_entities(self, text:str):
        """ Extract standard predefined entities using Spacy NER part of trf pipeline"""
        try:
            if not text or not text.strip():
                return []
            
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities, doc
        except Exception as e:
            logging.warning(f"spaCy NER Extraction failed: {e}")
            raise CustomException(e, sys)
    
    def extract_gliner_entities(self, text:str, labels:List[str]=None, threshold: float = 0.5):
        """ Extracts custom entities by zero-shot semantic matching using GLiNER """
        try:
            if not text or not text.strip():
                return []
            
            entities = self.gliner_model.extract_entities(text, labels, threshold,
                include_confidence=True, include_spans=True)
            return entities
        except Exception as e:
            logging.warning(f"GLiNER Extraction failed: {e}")
            raise CustomException(e, sys)
        
        
    def extract_keywords(self, query:str, top_n:int = 10, 
        diversity:float = 0.5, n_gram_range:tuple = (1, 2), MMR:bool=True):
        """ Extracts semantic keywords using KeyBERT """
        try:
            if not query or not len(query.strip().split()) < 5:
                return []
            
            keywords = self.kw_model.extract_keywords(query, top_n=top_n, 
                diversity=diversity, n_gram_range=n_gram_range, use_mmr=MMR, stop_words="english",
            )
            
            return keywords
        except Exception as e:
            logging.warning(f"Keyword Extraction failed: {e}")
            raise CustomException(e, sys)
        
    def process_articles(self, article:str, custom_labels:List[str]=None):
        """ Runs the full extraction pipeline for a single article """
        try:
            logging.info("Processing article for entities and keywords...")
            
            spacy_ent = self.extract_spacy_entities(article)
            gliner_ent = self.extract_gliner_entities(article, labels=custom_labels)
            keywords = self.extract_keywords(article)
            
            return {
                "Keywords": keywords,
                "predefined_ent": spacy_ent,
                "custom_ent": gliner_ent
            }
        except Exception as e:
            logging.warning(f"Article processing failed: {e}")
            raise CustomException(e, sys)
        
    
    def Visualize(self, article:str, type:str=None):
        """
        Visualizes article for entities and keywords usingHTML rendering.
        Compatible with spacy and GLiNER
        """
        try:
            logging.info("Visualizing article for entities and keywords...")
            
            if not type:
                raise ValueError("Type from model must be provided for visualization.")
            
            if type=='spacy':
                html = spacy.displacy.render(self.nlp(article), style="ent", page=True, jupyter=False)
                return html
            elif type=='gliner':
                html = GLiNER2.visualize(self.gliner_model, article)
                return html
            
        except Exception as e:
            logging.warning(f"Article visualization failed: {e}")
            raise CustomException(e, sys)
        
        
        
    
