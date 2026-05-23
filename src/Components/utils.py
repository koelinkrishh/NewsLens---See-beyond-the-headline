from typing import Tuple, Optional, List
import pandas as pd
import re, html, unicodedata
from bs4 import BeautifulSoup
from pydantic import BaseModel, field_validator, ValidationError
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer


from src.exception import CustomException
from src.logger import logging
from src.config import *


# Parameters
model_name = "sshleifer/distilbart-cnn-12-6"
max_tokens = 500

## Spacy pipeline
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    logging.warning(f"SpaCy model {SPACY_MODEL} not found. Attempting to download dynamically...")
    try:
        from spacy.cli import download
        download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
    except Exception as e:
        logging.error(f"Failed to download SpaCy model {SPACY_MODEL}: {e}")
        nlp = None

models = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
tokenizer = models.tokenizer # AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

class ChunkRequest(BaseModel):
    text: str
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v.strip() or not isinstance(v, str):
            raise ValidationError("Text must be a non-empty string")
        return v.strip()

def clean_text(s:str)->str:
    """
    Perform safe, canonical text cleaning for NLP tasks.
    Preserves linguistic structure.
    """
    try:
        if pd.isna(s) or not isinstance(s, str):
            return ''

        # 1. Fix broken encoding and HTML entities
        s = html.unescape(s)
        # 2. normalize unicode (NFKC helps)
        s = unicodedata.normalize('NFKC', s)
        # 3. Remove HTML tags (robust)
        s = BeautifulSoup(s, 'lxml').get_text(separator=" ")
        # 4. remove ZERO WIDTH and BOM chars
        s = re.sub(r'[\u200B-\u200D\uFEFF]', '', s)
        # 5. Normalize whitespace (spaces, tabs)
        s = re.sub(r"[ \t]+", " ", s)
        # 6. Remove repeated newlines
        s = re.sub(r"\n\s*\n+", "\n", s)
        # 7. Strip leading and trailing whitespace
        s = s.strip()

        return s
    except Exception as e:
        # Log warning but don't stop entire pipeline for one bad string
        logging.warning(f"Error cleaning text: {str(e)}")
        return ""

def split_sentence(text:str) -> List[str]:
    """
    Basic sentence splitting using SpaCy with a safe regex-based fallback if SpaCy is unavailable.
    """
    if nlp is None:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    doc = nlp(text)
    sentences = [str(sent).strip() for sent in doc.sents if str(sent).strip()]
    return sentences

