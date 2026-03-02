import os

# =========================================================
# PROJECT ROOT
# =========================================================
# config.py is inside src/, so go one level up
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# =========================================================
# SOURCE & COMPONENT PATHS
# =========================================================
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
COMPONENTS_DIR = os.path.join(SRC_DIR, "Components")
FEATURE_GENERATION_DIR = os.path.join(COMPONENTS_DIR, "1) FeatureGeneration.py")
EMBEDDING_DIR = os.path.join(COMPONENTS_DIR, "2) Embedding.py")
CLUSTERING_DIR = os.path.join(COMPONENTS_DIR, "3) SemanticClustering.py")
TOPIC_MODELING_DIR = os.path.join(COMPONENTS_DIR, "4) SemanticTopicModel.py")
SEARCH_FAISS_DIR = os.path.join(COMPONENTS_DIR, "5) SearchFAISS.py")
SUMMARIZATION_DIR = os.path.join(COMPONENTS_DIR, "6) Summarization.py")


# =========================================================
# DATA DIRECTORIES
# =========================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "Clean")

# =========================================================
# RAW DATA FILES
# =========================================================
DATA_SAMPLE_CSV = os.path.join(DATA_DIR, "data_sample.csv")

# =========================================================
# CLEAN / PROCESSED DATA FILES
# =========================================================
CLEANED_DATASET_PARQUET = os.path.join(CLEAN_DATA_DIR, "1) data_sample.parquet")
EMBEDDED_DATASET = os.path.join(CLEAN_DATA_DIR, "2) article_embeddings.parquet")
CLUSTER_DATASET = os.path.join(CLEAN_DATA_DIR, "3) article_clusters.parquet")

FINAL_DATASET_PARQUET = os.path.join(CLEAN_DATA_DIR, "4) Complete_dataset.parquet")

# =========================================================
# ARTIFACT / MODEL OUTPUT DIRECTORIES
# =========================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, "Models")

KMEANS_MODEL_DIR = os.path.join(MODEL_DIR, "kmeans_clusterer.joblib")
KMEANS_VECTORIZER = os.path.join(MODEL_DIR, "kmeans_vectorizer.joblib")
KMEANS_LABELS = os.path.join(MODEL_DIR, "kmeans_labels.pkl")
BERTOPIC_MODEL_DIR = os.path.join(MODEL_DIR, "topic_model.joblib")
BERTOPIC_MODEL_PARAMETERS = os.path.join(MODEL_DIR, "topic_model_parameters")

# --- NER + Keywords ---
Spacy_trf = "en_core_web_trf"
# Flair_ner = "flair/ner-english-ontonotes-fast"
Gliner_ner = "fastino/gliner2-large-v1" # OR "urchade/gliner_medium-v2.1"
Keybert_model = "all-MiniLM-L6-v2"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

# =========================================================
# NLP MODELS / SETTINGS
# =========================================================
SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5" # "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Load custom parameters from params.yaml without removing defaults
try: ## LOAD ALL MODEL CONFIG from yaml file
    import yaml
    PARAMS_FILE = os.path.join(PROJECT_ROOT, "params.yaml")
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "r") as f:
            _params = yaml.safe_load(f) or {}
        
        if "ner" in _params:
            Spacy_trf = _params["ner"].get("spacy_model_name", Spacy_trf)
            Gliner_ner = _params["ner"].get("gliner_model_name", Gliner_ner)
            Keybert_model = _params["ner"].get("keybert_model_name", Keybert_model)
        
        if "summarization" in _params:
            SUMMARIZATION_MODEL = _params["summarization"].get("model_name", SUMMARIZATION_MODEL)
            
        if "semantic_topic_model" in _params:
            SENTENCE_TRANSFORMER_MODEL = _params["semantic_topic_model"].get("embedding_model_name", SENTENCE_TRANSFORMER_MODEL)
            
        if "feature_generation" in _params:
            SPACY_MODEL = _params["feature_generation"].get("spacy_model", SPACY_MODEL)
except ImportError:
    print("Warning: PyYAML not installed. Using default parameters in config.py.")
except Exception as e:
    print(f"Error reading params.yaml: {e}")


# =========================================================
# DEBUG MODE
# =========================================================
if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("CLEAN_DATA_DIR:", CLEAN_DATA_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("Final dataset: ", FINAL_DATASET_PARQUET)
