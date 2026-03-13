# NewLens - Unsupervised News Intelligence Platform

NewLens is an end-to-end, high-performance NLP application designed to analyze news articles using unsupervised techniques. It transforms raw text into structured insights, providing linguistic heuristics, abstractive summarization, semantic clustering, and named entity recognition through a sleek, dual-process architecture (FastAPI + Streamlit).

---

## 🎯 Key Features

### 🔍 Linguistic Heuristics & Analysis
- **Grammar DNA:** Visualize POS distribution (Nouns, Verbs, Adjectives, etc.).
- **Sentence Pacing:** Interactive charts showing word count flow across sentences.
- **Readability & Density:** Automated Flesch Reading Ease scoring and information-to-fluff ratios.

### ✂️ AI-Powered Summarization
- **Abstractive Summary:** Leverages transformer-based models to compress long articles while maintaining semantic meaning.
- **Compression Control:** Adjust the summary length dynamically via the UI.

### 🧠 Semantic Topic Modeling & Clustering
- **BERTopic Integration:** Automatic topic discovery and probability distribution.
- **KMeans Clustering:** Maps articles to predefined clusters with c-TF-IDF keyword extraction.
- **Fit Analysis:** Visual gauges measuring the article's distance to its cluster centroid.

### 🏷️ Multi-Engine NER & Keywords
- **SpaCy:** Standard entity extraction (Persons, Orgs, Locations).
- **GLiNER:** Zero-shot Named Entity Recognition for custom labels.
- **KeyBERT:** Semantic keyword extraction with importance scoring.

### 🔗 Similar Article Retrieval (FAISS)
- **Vector Search:** Uses FAISS indexing to find top-K similar articles from a massive dataset.
- **Smart Routing:** Finds related content based on semantic embeddings, independent of source labels.

---

## 🏗️ System Architecture

The project follows a decoupled **Client-Server architecture** to ensure scalability and separation of concerns:

1.  **Backend (FastAPI):** Serves as the AI Engine. Loads heavy transformer models (Summarizers, Embeddings, NER) and exposes RESTful endpoints.
2.  **Frontend (Streamlit):** A premium, interactive dashboard that communicates with the API to visualize data and plots.
3.  **Vector Store (FAISS):** Efficient similarity search on compressed vector embeddings.

---

## 📂 Project Structure

```plaintext
News categorization/
├── app/
│   ├── api.py                # FastAPI Backend (The AI Engine)
│   └── app.py                # Streamlit Frontend (The Dashboard)
├── src/
│   ├── Components/           # Modular NLP Logic (Summarization, NER, Clustering, etc.)
│   ├── config.py             # Centralized paths and hyperparameters
│   ├── logger.py             # Custom logging system
│   └── exception.py          # Centralized error handling
├── Dataset/                  # Raw and processed datasets (Parquet/FAISS)
├── Models/                   # Saved model weights and parameters
├── Notebooks/                # EDA and Experimentation logs
├── requirements.txt          # Complete dependency list
└── README.md                 # Project documentation
```

---
## ▶️ Complete Setup & Execution Route

## ▶️ Setup & Installation

### Step 1. Environment & Dependencies
Open a terminal in the root project directory:
```bash
# Clone the Repository
git clone https://github.com/yourusername/NewsLens.git
cd NewsLens

# Create and activate environment
python -m venv .NewsEnv
.\.NewsEnv\Scripts\activate

# Install all necessary AI libraries
pip install -r requirements.txt
```

### Step 2. Download Transformer Models
Run the downloader to pull models locally (prevents downloading large files inside the Docker container):
```bash
python download_models.py
```

### Step 3. Train & Process Pipelines
Run the component scripts sequentially to process data and generate local artifacts (`Dataset/` and `Models/`):
```bash
python "src/Components/1) FeatureGeneration.py"
python "src/Components/2) Embedding.py"
python "src/Components/3) SemanticClustering.py"
python "src/Components/4) SemanticTopicModel.py"
python "src/Components/5) SearchFAISS.py"
```

### Step 4. Launch Backend AI Engine (via Minimal Docker Container)
The backend runs in an optimized, slim Docker container. It handles all the heavy model inference while mounting your local `Models/` and `Dataset/` directories.
```bash
# Build and run the Backend API
docker compose -f docker/docker-compose.yaml up --build -d
```
*Verify the API is online: [http://localhost:8000/](http://localhost:8000/)*

### Step 5. Launch Frontend Dashboard (Local Engine)
The frontend is run locally to provide a snappy experience. In your active `.NewsEnv` terminal:
```bash
# Install frontend-specific dependencies
pip install -r app/requirements-app.txt

# Launch Streamlit
streamlit run app/app.py
```
*Access the beautiful dashboard at [http://localhost:8501](http://localhost:8501)*

---

## 🔐 API Security & Access
The platform includes an AI Engine protection layer to prevent unauthorized access and resource abuse:
- **Default API Key:** `newslens_secret_key_2026`
- **Input Limits:** Max 20,000 characters per request (configured in `src/config.py`).
- **Authentication:** All AI-powered endpoints require an `X-API-Key` header.
- **Frontend Usage:** Upon launching the Streamlit app, enter the API key in the sidebar under **🔐 API Key:** to unlock the analysis modules.

---

## 📈 Roadmap & Extensions

- **Dockerization:** Containerizing the API and UI using Docker Compose for one-click deployment. [Complete]
- **[Planned] Cloud Deployment:** Hosting the API on AWS/GCP and the UI on Streamlit Cloud.
- **Temporal Trend Analysis:** Visualizing how specific news topics evolve over time.
- **Multilingual Support:** Expanding NER and Summarization to support non-English news sources.
- **RAG-based Article Q&A:** Integrating a Chat-over-Document feature using local LLMs.
- **Real-time News Ingestion:** Building a scraper to analyze live news feeds via RSS.

---

_👤 **Author**_
**Krishan Verma**
*Aspiring Data Scientist / ML Engineer*


"""
Docker command to build and run docker image

1. Build the docker image
```
docker compose -f docker/docker-compose.yaml build
```

2. Run the docker image
```
docker compose -f docker/docker-compose.yaml up -d
```

3. Stop the docker image
```
docker compose -f docker/docker-compose.yaml down
```

4. Verify the API is online
```
curl -X GET "http://localhost:8000/"
```

"""
