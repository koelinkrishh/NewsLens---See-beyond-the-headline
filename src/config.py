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



# FAISS_INDEX_DIR = os.path.join(ARTIFACTS_DIR, "faiss")
# EMBEDDING_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "embeddings")
# TOPIC_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "topic_models")
# LOG_DIR = os.path.join(ARTIFACTS_DIR, "logs")

# =========================================================
# NLP MODELS / SETTINGS
# =========================================================
SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5" # "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"


# =========================================================
# DEBUG MODE
# =========================================================
if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("CLEAN_DATA_DIR:", CLEAN_DATA_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("Final dataset: ", FINAL_DATASET_PARQUET)
