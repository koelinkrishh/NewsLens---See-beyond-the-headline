
# NewLens - Unsupervised News Article Analyzer

An end-to-end Single point NLP interfrace that analyzes any news article using unsupervised techniques to extract insights such as linguistic statistics, named entities, summaries, and similar articles — all wrapped inside a user-friendly frontend.

This project demonstrates practical NLP skills, ML engineering practices, and system design thinking, making it suitable for Data Scientist / ML Engineer entry-level roles.

### 🚀 Project Overview

Given a large news article, the system performs:
1. Text heuristics & linguistic analysis
2. Named Entity Recognition (NER) & keyword extraction
3. Abstractive text summarization
4. Semantic similarity search to find related articles
5. Interactive visualization through a web UI

## 🎯 Key Features
#### 🔍 Text Heuristics
- Word count
- Character count
- Sentence count
- Unique word ratio
- Lexical diversity
- Average sentence length

#### 🏷️ Entity & Keyword Extraction
1. Extracts:
- Persons
- Organizations
- Locations (GPE)
2. Stores entities in sets for future matching

3. Enables downstream similarity & analytics

#### ✂️ Text Summarization
* Generates ~100 word summaries
* Handles long documents via chunking
* Uses transformer-based pre-trained models

#### 🔗 Similar Article Retrieval
- Converts articles into vector embeddings
- Computes semantic similarity
- Returns Top-3 most similar articles from a large corpus
- Independent of article category labels

#### 🧠 Engineering Best Practices
- Centralized logging
- Custom exception handling
- Config-driven parameters
- Input validation
- Reproducible experiments

### Project Structure
``` plaintext
News categorization/
│
├── src/
│   ├── Components/                   (contains submodules)
│       ├── 
│   ├── logger.py                     ✅ central logging setup
│   ├── exception.py                  ✅ custom exception class
│   ├── config.py                     ✅ (stores paths, constants, hyperparameters)
│   └── utils.py                      ⚪ (helper utilities like plotting, validation)
│
├── Dataset/
|   ├── Data files                    (Raw data files)
│   ├── processed/                    (cleaned + combined parquet)
│   ├── intermediate/                 (feature datasets)
├── Models/                       (trained model weights/model)
│
├── notebooks/                        (EDA, experimentation)
├── requirements.txt
├── README.md
└── app/                              (will hold FastAPI or Streamlit app)
```

### 📊 Dataset
##### MN-DS-News Dataset
A real-world news dataset containing ~11,000 articles.

Key Columns:
1. date → Publication date as int count
2. source → News source
3. title → Headline
4. content → Main article text
5. author → Author (optional)
6. url → Original article link
7. category_level_1 / 2 → Used only for sanity checks (not training)

### 🧠 System Architecture
```
User Article
     ↓
Input Validation
     ↓
Text Preprocessing
     ↓
+-----------------------------+
|  Heuristics Analysis        |
|  NER & Keyword Extraction   |
|  Text Summarization         |
|  Embedding Generation       |
+-----------------------------+
     ↓
Similarity Search
     ↓
Streamlit UI Visualization
```

### __▶️ How to Run the App__
1️⃣ Install dependencies
``` python
 pip install -r requirements.txt
```
2️⃣ Run Streamlit
``` python
 streamlit run app/app.py
```


### 📈 Possible Extensions
`a. Topic modeling (BERTopic)`  \
`b. Temporal news trend analysis`   \
`c. Multilingual support`   \
`d. Real-time news ingestion`   \
`e. RAG-based article Q&A`  \
`f.User personalization`



_👤 Author_

Krishan Verma \
Aspiring Data Scientist / ML Engineer




