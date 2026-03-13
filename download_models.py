import spacy
import os
import sys

# Append root directory to sys.path to easily import config
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# FORCE HuggingFace to download to D: drive (inside the project Models folder)
# This prevents it from filling up the C: drive (C:\Users\.../.cache/huggingface)
os.environ["HF_HOME"] = os.path.join(root_dir, "Models", "huggingface_cache")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from gliner2 import GLiNER2
from keybert import KeyBERT

try:
    from src.config import (
        Spacy_trf, Gliner_ner, Keybert_model,
        SUMMARIZATION_MODEL, SENTENCE_TRANSFORMER_MODEL, SPACY_MODEL
    )
except ImportError:
    # Safe fallback if config missing for some reason
    Spacy_trf = "en_core_web_trf"
    Gliner_ner = "fastino/gliner2-large-v1"
    Keybert_model = "all-MiniLM-L6-v2"
    SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
    SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5"
    SPACY_MODEL = "en_core_web_sm"

def download_huggingface_models():
    print("--- Downloading Hugging Face Models ---")
    print(f"Downloading Summarization model: {SUMMARIZATION_MODEL}")
    AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
    AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
    
    print(f"Downloading Sentence Transformer: {SENTENCE_TRANSFORMER_MODEL}")
    SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    
    print(f"Downloading GLiNER2 model: {Gliner_ner}")
    GLiNER2.from_pretrained(Gliner_ner)
    
    print(f"Downloading KeyBERT model: {Keybert_model}")
    KeyBERT(model=Keybert_model)

def download_spacy_models():
    print("--- Downloading SpaCy Models ---")
    for model in [SPACY_MODEL, Spacy_trf]:
        try:
            print(f"Downloading SpaCy model: {model}")
            spacy.cli.download(model)
        except Exception as e:
            print(f"Error downloading {model}: {e}")

if __name__ == "__main__":
    download_spacy_models()
    download_huggingface_models()
    print("--- All core models downloaded successfully ---")
    
    
"""
Pip notes:
1. Add all requirements
pip freeze > requirements.txt

2. Add only missing files
# This gets your current environment and appends only what's NOT already in the file
pip freeze | Select-String -NotMatch (Get-Content requirements.txt -Raw) >> requirements.txt

"""
