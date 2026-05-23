# NewLens v2.0 - Unsupervised News Intelligence Platform (LangChain Edition)

NewLens is an end-to-end, high-performance NLP application designed to analyze news articles using unsupervised techniques. Modernized in **v2.0**, it runs on a fully decoupled **LangChain-powered architecture** that transforms raw text into structured insights, providing linguistic heuristics, abstractive summarization, named entity recognition, conversational RAG, and vector similarity recommendations through a sleek, dual-process containerized architecture (**FastAPI + Streamlit**).

---

## 🎯 Key Features

### 🔍 Linguistic Heuristics & Analysis (`FeatureGeneration_lc.py` & `Visualization_lc.py`)
- **Grammar DNA:** Visualize POS distribution (Nouns, Verbs, Adjectives, etc.) with custom Plotly donut charts.
- **Sentence Pacing:** Evaluate pacing and readability flow across sentences using interactive scatter plots with trendlines.
- **Readability Gauges:** Real-time dials indicating Flesch Reading Ease, Gunning Fog, and Kincaid Grade scores using colored gauges.

### ✂️ AI-Powered Summarization (`Summarization_lc.py` & `Visualization_lc.py`)
- **Abstractive Summary:** Leverages Hugging Face / Groq Llama-3 endpoints to compress long articles while maintaining core semantic meaning.
- **Compression Dial:** Dials showing exactly how effectively the AI summarizer compressed the text, alongside length comparison charts.

### 💬 Retrieval-Augmented Generation (RAG) (`QAchain.py`)
- **Chat Over Documents:** Real-time, conversational QA interface over the ingested article.
- **Vector Retrieve:** Chunking and vector similarity retrieval using either local FAISS index or PostgreSQL PGVector.

### 🏷️ Named Entity Recognition & KeyBERT (`NER_lc.py` & `Visualization_lc.py`)
- **Groq Llama-3 NER:** Deep Entity Extraction (Persons, Orgs, Locations, Events) with zero-shot capabilities.
- **KeyBERT Keywords:** Semantic keyword extraction mathematically scored and rendered on custom bar charts.
- **Highlight Engine:** HTML-highlighted article text highlighting all discovered entities dynamically.

### 🔗 Similar Article Recommendations (`Recommendation.py`)
- **Postgres PGVector Recommender:** A native PostgreSQL vector database storage and recommendation system.
- **Auto-Ingestion Pipeline:** On startup, the system reads your Parquet dataset file, automatically calculates the vector embeddings, and stores them permanently inside your Postgres database!

---

## 🏗️ System Architecture

The project follows a decoupled **Client-Server architecture** optimized for CPU efficiency and containerization:

1. **Backend API (FastAPI):** Serves as the AI Engine. Wired to the LangChain workflow, it exposes RESTful endpoints on port `8099`.
2. **Frontend UI (Streamlit):** Premium interactive visualizer on port `8501`. Connects to the FastAPI backend and uses Plotly to render beautiful data insights.
3. **Database (PostgreSQL + PGVector):** Acts as the vector database for storing article content, metadata, and 384-dimensional BAAI small embeddings.

---

## 📂 Project Structure

```plaintext
News categorization/
├── app/
│   ├── api_lc.py             # LangChain FastAPI Backend (The AI Engine)
│   └── app_lc.py             # Streamlit Frontend Dashboard (The Visualizer)
├── docker/
│   ├── Dockerfile.api        # CPU-only optimized FastAPI Dockerfile (with pre-baked NLP)
│   ├── Dockerfile.app        # Streamlit App Dockerfile
│   └── docker-compose.yaml   # API + App Orchestration with PostgreSQL volume mappings
├── src/
│   ├── Components/           # Modular LangChain-based Logic
│   │   ├── Embedding_lc.py   # Truly lazy Embedding Engine (bge-small-en-v1.5)
│   │   ├── FeatureGeneration_lc.py # Heuristic feature counts (POS, readability)
│   │   ├── NER_lc.py         # Named Entities (Groq) & Keywords (KeyBERT)
│   │   ├── QAchain.py        # Conversational QA / RAG Engine
│   │   ├── Recommendation.py # PostgreSQL PGVector similarity recommender
│   │   ├── Summarization_lc.py # Abstractive summarization via LLM (Groq)
│   │   └── Visualization_lc.py # Plotly gauges, sentence flow charts, and donut metrics
│   ├── config.py             # Centralized system configurations and credentials
│   ├── logger.py             # Custom logging system (outputs to logs/ folder)
│   └── exception.py          # Unified exception handler
├── Dataset/                  # Parquet files and raw datasets
├── Models/                   # FAISS indexes and registry metadata
├── requirements-api.txt      # API package requirements (pinned CPU PyTorch, PGVector)
├── requirements-app.txt      # Streamlit package requirements (streamlit, plotly, charts)
└── README.md                 # Project documentation
```

---

## 🗄️ PostgreSQL Setup Guide

To run the recommendation module, you must have a running instance of PostgreSQL with the `pgvector` extension enabled. You can set this up in one of two ways:

### Option A: Using Docker (Highly Recommended)
This is the fastest, cleanest route. It spins up a Postgres 16 instance preloaded with the `pgvector` extension, requiring zero compilation or manual library setups:
```bash
docker run --name pgvector-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=root \
  -e POSTGRES_DB=news_rag \
  -p 5432:5432 -d pgvector/pgvector:pg16
```

### Option B: Local Setup (Native PostgreSQL)
If you already have PostgreSQL installed locally on your system:
1. **Download the pgvector Extension**:
   - On Windows: Download the pre-compiled `pgvector` binaries for your Postgres version or build it using MSVC.
   - On macOS: Run `brew install pgvector`.
   - On Linux: Run `sudo apt-get install postgresql-16-pgvector` (adjust version according to your local Postgres installation).
2. **Launch Postgres** and ensure it is listening on port `5432`.
3. **Configure Database**: Connect to your database server and create a database named `news_rag`:
   ```sql
   CREATE DATABASE news_rag;
   ```
*(Note: Your code automatically runs `CREATE EXTENSION IF NOT EXISTS vector;` when connecting, so you do not need to manually create the extension inside the database).*

---

## ▶️ Setup & Installation (Local Execution)

### Step 1. Environment & Dependencies
Open a terminal in the root project directory:
```bash
# Clone the Repository
git clone https://github.com/yourusername/NewsLens.git
cd NewsLens

# Create and activate environment
python -m venv .NewsEnv
.\.NewsEnv\Scripts\activate

# Install all necessary API & App dependencies
pip install -r requirements-api.txt
pip install -r requirements-app.txt
```

### Step 2. Setup Credentials (`.env`)
Create a `.env` file in the root project directory containing your API credentials and Postgres connection details:
```env
# AI Endpoints Keys
HUGGINGFACEHUB_API_TOKEN = "your_hf_token"
GROQ_API_KEY = "your_groq_key"
NEWSLENS_API_KEY = "newslens_secret_key_2026"

# PostgreSQL Configuration
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "root"
POSTGRES_HOST = "localhost"   # (Use host.docker.internal inside docker-compose)
POSTGRES_PORT = "5432"
POSTGRES_DB = "news_rag"
```

### Step 3. Launch App and API locally
In your active `.NewsEnv` shell, start the FastAPI engine:
```bash
python app/api_lc.py
```
In a separate active `.NewsEnv` shell, start the Streamlit visualizer dashboard:
```bash
streamlit run app/app_lc.py
```
*Access the beautiful dashboard at [http://localhost:8501](http://localhost:8501)*

---

## 🐳 Running with Docker Compose (Fully Orchestrated)

The platform is fully containerized. It includes a series of optimizations to prevent memory overflows and run with maximum performance on standard machines:
- **Truly Lazy Loading:** The vector embedding model and PyTorch libraries are initialized lazily. Startup memory drops to **under 150MB**, fully preventing Docker OOM crashes.
- **Pre-baked NLP Resources:** The SpaCy model (`en_core_web_sm`) and NLTK assets (`punkt`, `punkt_tab`, `stopwords`) are baked into the API image at build time, ensuring smooth, instant, and network-independent startup.
- **Dataset Volume Mapping:** The host's `Dataset/` directory is mounted dynamically as a read-only volume. This keeps the Docker images tiny (no need to copy hundreds of megabytes of raw files inside the container layer).
- **Auto-Ingestion Pipeline:** On first container boot, the API loads the dataset `News_dataset.parquet` which is configured in src/config.py as CLEANED_DATASET_PARQUET, computes embeddings for the articles, and ingests them directly into your PostgreSQL table.

### Step 1. Rebuild and Compile Container Images
From your project terminal, execute the build sequence:
```bash
docker compose -f docker/docker-compose.yaml build
```

### Step 2. Bring Container Stack Online
Start both API and UI services in the background:
```bash
docker compose -f docker/docker-compose.yaml up -d
```

### Step 3. Inspect Ingestion Status & Health Check
Verify the backend startup logs and see the database population occur in seconds:
```bash
# Check container statuses (API should be marked as healthy)
docker compose -f docker/docker-compose.yaml ps

# Print the startup logs inside the container
docker compose -f docker/docker-compose.yaml logs api
```

### Step 4. Tear Down Services
When done, you can stop the containers and free up resources:
```bash
docker compose -f docker/docker-compose.yaml down
```

---

## 📂 Using Your Own Custom News Dataset

You can easily replace our default news articles with **any other custom news dataset**:

1. **Format Your Dataset**: Clean your text rows and export it as an **Apache Parquet** file. The file must contain at least the following columns:
   - `Content`: The full text of the news articles.
   - `Summary` (Optional): A pre-computed short summary of the articles.
2. **Save the File**: Save your custom parquet file exactly at the path configured in `src/config.py`:
   - CLEANED_DATASET_PARQUET: `Dataset/Clean/News_dataset.parquet`
3. **Launch the Stack**: Boot the application locally or via `docker compose`.
4. **Automated Ingest & Vector Ingestion**: 
   - During boot, the system checks PostgreSQL. If the table contains fewer than 100 articles, it triggers the auto-ingestion process.
   - It reads your custom `News_dataset.parquet` file, segments the contents, generates 384-dimensional vector embeddings using the local `bge-small-en-v1.5` CPU-only SentenceTransformer, and inserts the articles into PostgreSQL.
   - If your custom Parquet file already contains an `embedding` column (list of floats or numpy array), your optimized engine will automatically detect it, skip the heavy CPU calculation, and bulk-load all rows in under 5 seconds!

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
- **RAG-based Article Q&A:** Integrating a Chat-over-Document feature using local LLMs. [Complete]
- **Persistent PGVector Database:** Moving recommendations out of volatile local frames to permanent vector stores. [Complete]
- **Startup Memory Optimizations:** Introducing lazy loaders to bring container boot footprint below 150MB. [Complete]
- **Temporal Trend Analysis:** Visualizing how specific news topics evolve over time.
- **Multilingual Support:** Expanding NER and Summarization to support non-English news sources.

---

_👤 **Author**_
**Krishan Verma**
*Aspiring Data Scientist / ML Engineer*
