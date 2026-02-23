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

## ▶️ Setup & Installation

Follow these steps to set up the environment and run the project from scratch.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/NewsLens.git
cd NewsLens
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv newsenv
# Windows:
.\newsenv\Scripts\activate
# Mac/Linux:
source newsenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run the Application
You need to run **two separate processes**:

**Step 4a: Start the Backend API (AI Engine)**
Open a terminal and run:
```bash
uvicorn app.api:app --reload
```
*Wait for the "Startup complete" message.*

**Step 4b: Start the Frontend Dashboard**
Open a second terminal (ensure the environment is active) and run:
```bash
streamlit run app/app.py
```

---

## 📈 Roadmap & Extensions

- **[Planned] Dockerization:** Containerizing the API and UI using Docker Compose for one-click deployment.
- **[Planned] Cloud Deployment:** Hosting the API on AWS/GCP and the UI on Streamlit Cloud.
- **Temporal Trend Analysis:** Visualizing how specific news topics evolve over time.
- **Multilingual Support:** Expanding NER and Summarization to support non-English news sources.
- **RAG-based Article Q&A:** Integrating a Chat-over-Document feature using local LLMs.
- **Real-time News Ingestion:** Building a scraper to analyze live news feeds via RSS.

---

_👤 **Author**_
**Krishan Verma**
*Aspiring Data Scientist / ML Engineer*
