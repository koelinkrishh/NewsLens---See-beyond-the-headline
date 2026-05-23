import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
from functools import lru_cache

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# KeyBERT for local Keyword Extraction & Semantic Scoring
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from src.config import Keybert_model

# Core system imports
from src.logger import logging
from src.exception import CustomException


# ---------------------------------------------------------------------------
# Pydantic structured output schemas
# ---------------------------------------------------------------------------
class ExtractedEntity(BaseModel):
    text: str = Field(description="The exact text span of the entity as it appears in the article")
    label: str = Field(description="The category label for this entity (e.g., Person, Organisation, Country, Date, Law, Technology, Medical)")


class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity] = Field(description="List of all extracted entities with their category labels")

# ---------------------------------------------------------------------------
# Label registry — union of SpaCy 18-class + full GLiNER label set
# ---------------------------------------------------------------------------
SPACY_LABELS = {
    "PERSON", "ORG", "GPE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "CARDINAL", "ORDINAL", "NORP", "FAC", "LOC",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
}

GLINER_LABELS = [
    "Person", "Celebrity", "Political party", "Politician", "Activist", "Criminal", "Victim", "Witness",
    "Profession", "Job title", "Author", "Scientist", "Journalist", "Speaker", "Writer", "Artist",
    "Affiliation", "Organisation", "Company", "Startup", "Institution", "College", "University",
    "Government agency", "Military organisation", "Union", "Sports team", "Media",
    "Country", "City", "State", "Region", "Continent", "Climate zone", "Forest", "Desert",
    "Mountain", "Park", "Water body",
    "Building", "Airport", "Monument", "Landmark",
    "Date", "Time", "Duration", "Percent", "Money", "Temperature", "Speed", "Age",
    "Law", "Case", "Judge", "Constitution", "Election",
    "Medical", "Disease", "Drug", "Symptom", "Chemical",
    "Location", "Product", "Event", "Work_of_art", "Language",
    "Business", "Market", "Stock", "Currency",
    "Sport", "Games", "Award",
    "Art", "Book", "Movie", "TV show",
    "Computer", "Vehicle", "Machine", "Programming language", "Technology",
    "Color", "Shape", "Size", "Weight", "Weapon", "Battle", "Natural disaster",
    "Quantity", "Ordinal", "Cardinal",
    "Animal", "Plant", "Organism",
]

# Color palette for visualization
LABEL_COLORS = {
    # People
    "Person": "#fce4ec", "Celebrity": "#fce4ec", "Politician": "#fce4ec",
    "Activist": "#fce4ec", "Criminal": "#fce4ec", "Victim": "#fce4ec",
    "Witness": "#fce4ec", "Author": "#fce4ec", "Scientist": "#fce4ec",
    "Journalist": "#fce4ec", "Speaker": "#fce4ec", "Writer": "#fce4ec", "Artist": "#fce4ec",
    # Organisations
    "Organisation": "#e3f2fd", "Company": "#e3f2fd", "Startup": "#e3f2fd",
    "Institution": "#e3f2fd", "College": "#e3f2fd", "University": "#e3f2fd",
    "Government agency": "#e3f2fd", "Military organisation": "#e3f2fd",
    "Union": "#e3f2fd", "Sports team": "#e3f2fd", "Media": "#e3f2fd",
    "Political party": "#e3f2fd", "Affiliation": "#e3f2fd",
    # Locations
    "Country": "#e8f5e9", "City": "#e8f5e9", "State": "#e8f5e9",
    "Region": "#e8f5e9", "Continent": "#e8f5e9", "Location": "#e8f5e9",
    "Forest": "#e8f5e9", "Desert": "#e8f5e9", "Mountain": "#e8f5e9",
    "Park": "#e8f5e9", "Water body": "#e8f5e9", "Climate zone": "#e8f5e9",
    "Building": "#e8f5e9", "Airport": "#e8f5e9", "Monument": "#e8f5e9", "Landmark": "#e8f5e9",
    # Temporal & Numeric
    "Date": "#fff3e0", "Time": "#fff3e0", "Duration": "#fff3e0",
    "Percent": "#fff3e0", "Money": "#fff3e0", "Quantity": "#fff3e0",
    "Cardinal": "#fff3e0", "Ordinal": "#fff3e0",
    "Temperature": "#fff3e0", "Speed": "#fff3e0", "Age": "#fff3e0",
    # Legal & Political
    "Law": "#f3e5f5", "Case": "#f3e5f5", "Judge": "#f3e5f5",
    "Constitution": "#f3e5f5", "Election": "#f3e5f5",
    # Medical & Science
    "Medical": "#e0f2f1", "Disease": "#e0f2f1", "Drug": "#e0f2f1",
    "Symptom": "#e0f2f1", "Chemical": "#e0f2f1",
    # Business & Finance
    "Business": "#fff9c4", "Market": "#fff9c4", "Stock": "#fff9c4", "Currency": "#fff9c4",
    # Technology
    "Technology": "#e0f7fa", "Computer": "#e0f7fa", "Vehicle": "#e0f7fa",
    "Machine": "#e0f7fa", "Programming language": "#e0f7fa",
    # Arts & Culture
    "Art": "#f1f8e9", "Book": "#f1f8e9", "Movie": "#f1f8e9", "TV show": "#f1f8e9",
    "Work_of_art": "#f1f8e9", "Language": "#f1f8e9",
    "Product": "#f1f8e9", "Event": "#f1f8e9", "Award": "#f1f8e9",
    # Sport
    "Sport": "#ede7f6", "Games": "#ede7f6",
    # Nature
    "Animal": "#dcedc8", "Plant": "#dcedc8", "Organism": "#dcedc8",
    # Misc
    "Profession": "#cfd8dc", "Job title": "#cfd8dc",
    "Color": "#cfd8dc", "Shape": "#cfd8dc", "Size": "#cfd8dc", "Weight": "#cfd8dc",
    "Weapon": "#cfd8dc", "Battle": "#cfd8dc", "Natural disaster": "#cfd8dc",
}

# Border color to match each background family
LABEL_BORDER_COLORS = {
    "#fce4ec": "#f8bbd0", "#e3f2fd": "#bbdefb", "#e8f5e9": "#a5d6a7",
    "#fff3e0": "#ffcc80", "#f3e5f5": "#ce93d8", "#e0f2f1": "#80cbc4",
    "#fff9c4": "#fff176", "#e0f7fa": "#80deea", "#f1f8e9": "#c5e1a5",
    "#ede7f6": "#b39ddb", "#dcedc8": "#aed581", "#cfd8dc": "#b0bec5",
}

DEFAULT_BG = "#e0f7fa"
DEFAULT_BORDER = "#b2ebf2"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Named Entity Recognition (NER) engine for news articles.

Your task:
1. Extract ALL meaningful named entities from the article text.
2. For each entity, provide the EXACT text span as it appears in the article and a category label.
3. Ensure valid JSON output without trailing or double commas.

Use these entity categories (prioritize the most specific label that applies):
{label_list}

Rules:
- Entity text must be an EXACT substring copied from the article — do NOT paraphrase or modify or pull synonym.
- Each entity should appear only once even if it occurs multiple times in the text.
- Do not change the format of entity - no synonym, no modification, no additional words, no expansion, no hallucination. Only exact exact words.
- Choose the most specific label available (e.g., "Politician" over "Person" if applicable)."""

HUMAN_PROMPT = """Article Text:
{article}"""

class InformationExtractor:
    """
    Modernized Information Extraction Engine built with LangChain & KeyBERT.
    Uses LLMs for Structured NER and local embedding models (bge-small) for math-based Keyword scoring.
    """
    def __init__(self,
        provider: str = "groq",
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None
    ):
        logging.info(f"Initializing InformationExtractor [Provider: {provider}, Model: {model}]...")
        
        try:
            self.provider = provider.lower()
            self.model = model
            
            # 1. Route to correct provider and validate API keys explicitly
            if self.provider == "groq":
                key = api_key or os.getenv("GROQ_API_KEY")
                if not key:
                    raise ValueError("GROQ_API_KEY is missing. You must provide a valid Groq API Key.")
                self.llm = ChatGroq(model=self.model, groq_api_key=key, temperature=0)
                
            elif self.provider == "openai":
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError("OPENAI_API_KEY is missing. You must provide a valid OpenAI API Key.")
                self.llm = ChatOpenAI(model=self.model, openai_api_key=key, temperature=0)
                
            elif self.provider == "huggingface":
                key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not key:
                    raise ValueError("HF_TOKEN / HUGGINGFACEHUB_API_TOKEN is missing. You must provide a valid Hugging Face Token.")
                self.llm = ChatOpenAI(
                    model=self.model,
                    openai_api_key=key,
                    base_url="https://router.huggingface.co/v1",
                    temperature=0
                )
            else:
                raise ValueError(f"Unsupported provider: '{self.provider}'. Choose from 'groq', 'openai', or 'huggingface'.")
            
            # 2. Build the structured extraction chain with ChatPromptTemplate
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ])
            
            # 3. Prepare KeyBERT lazy initialization to avoid heavy downloads at startup
            logging.info("KeyBERT will be lazily initialized on first keyword extraction.")
            self.kw_model = None
            
            logging.info("InformationExtractor initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize InformationExtractor: {str(e)}")
            raise CustomException(e, sys)
        
    @lru_cache(maxsize=128)
    def _get_llm_results(self, text: str, labels: tuple = None):
        """ Internal cached method to fetch raw structured entities from the LLM """
        active_labels = list(GLINER_LABELS)
        if labels:
            for lbl in labels:
                if lbl not in active_labels:
                    active_labels.append(lbl)
                    
        label_list_str = ", ".join(active_labels)
        formatted_messages = self.prompt_template.format_messages(label_list=label_list_str, article=text)
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        return structured_llm.invoke(formatted_messages)
    
    def extract_entities(self, text: str, labels: List[str] = None) -> List[dict]:
        """ Extracts all entities (both standard/predefined and custom) using LLM in a single run and scores them with KeyBERT """
        try:
            if not text or not text.strip():
                return []
            
            labels_tuple = tuple(labels) if labels is not None else None
            result = self._get_llm_results(text, labels_tuple)
            
            flat_entities = []
            seen = set()
            
            # Embed the full document using KeyBERT's underlying sentence transformer (lazy)
            doc_emb = None
            try:
                self._ensure_kw_model()
                if self.kw_model is not None:
                    doc_emb = self.kw_model.model.embed([text])
            except Exception as e:
                logging.warning(f"Failed to embed doc (KeyBERT lazy init): {e}")
                doc_emb = None
            
            for ent in result.entities:
                # Deduplicate by lowercase text and label
                dedup_key = (ent.text.lower(), ent.label.lower())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                
                start_idx = text.find(ent.text)
                if start_idx == -1:
                    continue  # Discard hallucinated entities perfectly
                    
                end_idx = start_idx + len(ent.text)
                
                # Calculate true Cosine Similarity score for the entity against the document using KeyBERT
                score = 1.0
                if doc_emb is not None:
                    try:
                        self._ensure_kw_model()
                        if self.kw_model is not None and doc_emb is not None:
                            ent_emb = self.kw_model.model.embed([ent.text])
                            score = max(0.0, float(cosine_similarity(doc_emb, ent_emb)[0][0]))
                    except Exception as e:
                        logging.warning(f"Failed to compute entity score: {e}")
                        pass
                        
                flat_entities.append({
                    'text': ent.text,
                    'label': ent.label,
                    'score': round(score, 4),
                    'start': start_idx,
                    'end': end_idx
                })
                
            return flat_entities
        
        except Exception as e:
            logging.warning(f"LLM Entity Extraction failed: {e}")
            raise CustomException(e, sys)
    
    def extract_spacy_entities(self, text: str):
        """ Extract standard predefined entities using LLM (replaces spaCy trf) """
        try:
            entities = self.extract_entities(text)
            spacy_ents = []
            for ent in entities:
                if ent['label'].upper() in SPACY_LABELS:
                    spacy_ents.append((ent['text'], ent['label']))
            return spacy_ents
        except Exception as e:
            logging.warning(f"LLM Predefined Extraction failed: {e}")
            raise CustomException(e, sys)
    
    def extract_gliner_entities(self, text: str, labels: List[str] = None, threshold: float = 0.5):
        """ Extracts custom entities using LLM (replaces GLiNER) and scores them mathematically with KeyBERT """
        try:
            entities = self.extract_entities(text, labels=labels)
            gliner_ents = []
            for ent in entities:
                if ent['label'].upper() not in SPACY_LABELS:
                    gliner_ents.append(ent)
            return gliner_ents
        except Exception as e:
            logging.warning(f"LLM Custom Extraction failed: {e}")
            raise CustomException(e, sys)

    def _ensure_kw_model(self):
        """Lazily initialize KeyBERT only when needed to avoid HF downloads at startup."""
        if self.kw_model is not None:
            return
        try:
            logging.info(f"Initializing KeyBERT model: {Keybert_model} ...")
            self.kw_model = KeyBERT(model=Keybert_model)
            logging.info("KeyBERT initialized successfully.")
        except Exception as e:
            logging.warning(f"Failed to initialize KeyBERT model ({Keybert_model}): {e}")
            self.kw_model = None
    
    @lru_cache(maxsize=128)
    def _get_keywords(self, query: str, top_n: int, diversity: float, n_gram_range: tuple, MMR: bool):
        """ Internal cached method for KeyBERT """
        # Ensure the KeyBERT model is ready; if not available, return empty list as graceful fallback
        self._ensure_kw_model()
        if not self.kw_model:
            logging.warning("KeyBERT not available; keyword extraction will return an empty list.")
            return []
        return self.kw_model.extract_keywords(
            query, top_n=top_n, diversity=diversity, keyphrase_ngram_range=n_gram_range,
            use_mmr=MMR, stop_words="english"
        )
    
    def extract_keywords(self, query: str, top_n: int = 10, diversity: float = 0.5, n_gram_range: tuple = (1, 2), MMR: bool = True):
        """ Extracts semantic keywords using KeyBERT """
        try:
            if not query or len(query.strip().split()) < 5:
                return []
            return self._get_keywords(query, top_n, diversity, tuple(n_gram_range), MMR)
        except Exception as e:
            logging.warning(f"Keyword Extraction failed: {e}")
            raise CustomException(e, sys)
    
    def process_articles(self, article: str, custom_labels: List[str] = None):
        """ Runs the full extraction pipeline for a single article """
        try:
            logging.info("Processing article for entities and keywords...")
            
            entities = self.extract_entities(article, labels=custom_labels)
            keywords = self.extract_keywords(article)
            
            spacy_ent = []
            gliner_ent = []
            for ent in entities:
                if ent['label'].upper() in SPACY_LABELS:
                    spacy_ent.append((ent['text'], ent['label']))
                else:
                    gliner_ent.append(ent)
            
            return {
                "keywords": keywords,
                "entities": entities,
                "predefined_ents": spacy_ent,
                "custom_ents": gliner_ent
            }
        except Exception as e:
            logging.warning(f"Article processing failed: {e}")
            raise CustomException(e, sys)
    
    def Visualize(self, article: str, type: str = 'gliner', entities: List[dict] = None) -> str:
        """ Generates HTML visualization compatible with spacy/GLiNER """
        try:
            logging.info("Visualizing article entities...")
            ents = entities if entities is not None else self.extract_gliner_entities(article)
            
            # Deduplicate overlapping spans — keep the longest span for each position
            valid_ents = sorted(ents, key=lambda x: (x['start'], -(x['end'] - x['start'])))
            deduped = []
            last_end = -1
            for ent in valid_ents:
                if ent['start'] >= last_end:
                    deduped.append(ent)
                    last_end = ent['end']
                    
            sorted_ents = sorted(deduped, key=lambda x: x['start'], reverse=True)
            
            html = article
            for ent in sorted_ents:
                start, end, label = ent['start'], ent['end'], ent['label']
                entity_text = html[start:end]
                bg_color = LABEL_COLORS.get(label, DEFAULT_BG)
                border_color = LABEL_BORDER_COLORS.get(bg_color, DEFAULT_BORDER)
                
                mark_str = (
                    f'<mark style="background: {bg_color}; padding: 0.15em 0.3em; '
                    f'margin: 0 0.1em; border-radius: 0.35em; font-weight: 600; '
                    f'border: 1px solid {border_color}; display: inline-block;">'
                    f'{entity_text} '
                    f'<span style="font-size: 0.65em; margin-left: 0.3em; '
                    f'color: #37474f; font-variant: small-caps; font-weight: 700; '
                    f'vertical-align: middle;">{label}</span>'
                    f'</mark>'
                )
                html = html[:start] + mark_str + html[end:]
            
            html = html.replace('\n', '<br>')
            return (
                f'<div style="line-height: 2.2; font-size: 1.1em; '
                f'font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', sans-serif; padding: 1em; '
                f'background: #fafafa; border-radius: 0.5em; border: 1px solid #e0e0e0;">'
                f'{html}</div>'
            )
        except Exception as e:
            logging.warning(f"Failed to generate visualization HTML: {e}")
            raise CustomException(e, sys)
    
    def VisualizePlotly(self, article: str, entities: List[dict] = None):
        """ Plotly Bar Chart and Table representation for Streamlit """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            ents = entities if entities is not None else self.extract_gliner_entities(article)
            if not ents:
                fig = go.Figure()
                fig.update_layout(title="No Entities Found")
                return fig
                
            df = pd.DataFrame(ents)
            freq = df.groupby(['label', 'text']).size().reset_index(name='count')
            freq = freq.sort_values(by='count', ascending=True)
            
            fig = make_subplots(
                rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "table"}]],
                column_widths=[0.6, 0.4], subplot_titles=("Entity Frequency", "Extracted Entities List")
            )
            
            for label in freq['label'].unique():
                subset = freq[freq['label'] == label]
                color = LABEL_COLORS.get(label, DEFAULT_BG)
                fig.add_trace(go.Bar(
                    y=subset['text'] + " (" + subset['label'] + ")", x=subset['count'], name=label,
                    orientation='h', marker_color=color,
                    marker_line_color=LABEL_BORDER_COLORS.get(color, DEFAULT_BORDER), marker_line_width=1.5
                ), row=1, col=1)
                
            df_table = df[['text', 'label']].drop_duplicates().sort_values(by='label')
            fig.add_trace(go.Table(
                header=dict(values=["Entity Text", "Label Category"], fill_color="#263238", font=dict(color="white", size=14), align="left"),
                cells=dict(values=[df_table['text'], df_table['label']], fill_color="#f5f5f5", font=dict(color="#212121", size=12), align="left", height=25)
            ), row=1, col=2)
            
            fig.update_layout(title_text="Named Entity Extraction Results", height=600, showlegend=False, plot_bgcolor="white")
            fig.update_xaxes(title_text="Mentions", row=1, col=1)
            return fig
            
        except Exception as e:
            logging.warning(f"Failed to generate Plotly visualization: {e}")
            raise CustomException(e, sys)
    

# --- Local Verification script ---
if __name__ == "__main__":
    print("Testing Clean LangChain InformationExtractor...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv("GROQ_API_KEY"):
            print("Checking missing key exception handling...")
            try:
                extractor = InformationExtractor(provider="groq")
            except Exception as ex:
                print(f"[SUCCESS] Correctly caught missing key exception: {ex}")
        else:
            extractor = InformationExtractor(provider="groq")
            test_article = (
                "Microsoft Corp. CEO Satya Nadella announced new AI cloud integration services "
                "at the Microsoft Build 2026 conference in Seattle, Washington today. "
                "The event showcased Azure OpenAI partnerships and a $2 billion investment in India."
            )
            
            print("\nRunning extraction...")
            results = extractor.process_articles(test_article)
            
            print("\nKeywords (KeyBERT with Math Scores):")
            for kw, score in results["keywords"]:
                print(f"- {kw}: {score:.4f}")
            
            print("\nPredefined Entities (SpaCy style):", results["predefined_ents"])
            
            print("\nCustom Entities (Scored with KeyBERT Semantic Similarity):")
            for ent in results["custom_ents"]:
                print(f" - {ent['text']} [{ent['label']}] (Score: {ent['score']})")
            
            print("\nGenerating interactive Plotly visualization...")
            fig = extractor.VisualizePlotly(test_article, entities=results["custom_ents"])
            fig.show()
            
            print("\n[SUCCESS] NER VERIFICATION SUCCESSFUL!")
            
    except Exception as e:
        print(f"Test run encountered an issue: {e}")
