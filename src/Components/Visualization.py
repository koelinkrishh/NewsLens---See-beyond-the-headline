"""
Inference visualization components for the News Article Analyzer.
Each class: loads its model (or receives precomputed data) and returns Plotly figures
and data for frontend inference analysis.
"""
import os
import re
import sys
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode, ColumnsAutoSizeMode

from src.logger import logging
from src.exception import CustomException
from src.config import (
    FINAL_DATASET_PARQUET,
    BERTOPIC_MODEL_PARAMETERS,
    SENTENCE_TRANSFORMER_MODEL,
)
from src.Components.SemanticClustering import KMeansTopicLabeler

# 1. Suppress the oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 2. Suppress other TensorFlow logging (0=all, 1=no INFO, 2=no INFO/WARN, 3=no ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ---------------------------------------------------------------------------
# Helpers - > Seperate as it is needed in most of classes
# ---------------------------------------------------------------------------
def _sentences_from_text(text: str) -> List[str]:
    """Split text into sentences (regex-based, no heavy NLP)."""
    if not text or not str(text).strip():
        return []
    text = str(text).strip()
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.split()) > 0]


# ============================================================================
# 1) NewsVisualizer — Heuristic features & readability
# ============================================================================
class NewsVisualizer:
    """
    Inference visualization for heuristic features (length, POS, NER counts, readability).
    Accepts a single-row DataFrame or Series with feature columns; returns Plotly figures.
    """
    POS_COLUMNS = ['noun_count', 'verb_count', 'adj_count', 'adv_count', 'pronoun_count']
    POS_LABELS = ['Nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Pronouns']

    def __init__(self):
        logging.info("Initializing NewsVisualizer Engine...")
        self.colors = px.colors.qualitative.Prism

    def _normalize_input(self, article_data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        if isinstance(article_data, pd.DataFrame):
            return article_data.iloc[0]
        return article_data

    def _safe_get(self, data: pd.Series, key: str, default: Any = 0) -> Any:
        """Resolve column name variants (e.g. stopword_Count vs stopword_count)."""
        if key in data.index:
            return data[key]
        alt = key.replace('_count', '_Count') if '_count' in key else key.replace('_Count', '_count')
        return data.get(alt, default)

    # 1. GRAMMAR DNA: Internal POS Composition
    def plot_grammar_composition(self,
        article_data: Union[pd.Series, pd.DataFrame],
        plot_type: str = 'bar', height: int = 380,
    ) -> go.Figure:
        """POS composition: bar or donut."""
        try:
            data = self._normalize_input(article_data)
            pos_counts = [max(0, int(data.get(c, 0))) for c in self.POS_COLUMNS]
            df_plot = pd.DataFrame({'POS': self.POS_LABELS, 'Count': pos_counts})
            df_plot = df_plot[df_plot['Count'] > 0]
            if df_plot.empty:
                logging.info("Article has no POS counts; using zeros.")
                df_plot = pd.DataFrame({'POS': self.POS_LABELS, 'Count': [0] * len(self.POS_LABELS)})

            if plot_type == 'pie':
                fig = px.pie(
                    df_plot, values='Count', names='POS',
                    title="Grammatical Composition",
                    hole=0.5, height=height, color_discrete_sequence=self.colors,
                )
                fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
            else:
                fig = px.bar(
                    df_plot, x='POS', y='Count',
                    title="Grammatical Composition", text='Count',
                    color='POS', color_discrete_sequence=self.colors, height=height,
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, xaxis_tickangle=-30, xaxis=dict(automargin=True))
            fig.update_layout(margin=dict(t=50, b=50, l=50, r=50), template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white')
            
            return fig
        except Exception as e:
            logging.error(f"plot_grammar_composition failed: {e}")
            raise CustomException(e, sys)

    # 2. STRUCTURAL FLOW: Sentence Rhythm
    def plot_sentence_flow(self,
        article_data: Union[pd.Series, pd.DataFrame],
        height: int = 340,
    ) -> go.Figure:
        """Sentence pacing: word count per sentence."""
        try:
            data = self._normalize_input(article_data)
            text = str(data.get('Content', ''))
            sentences = _sentences_from_text(text)
            
            if not sentences:
                fig = go.Figure()
                fig.add_annotation(text="No sentences to display", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig.update_layout(height=height, title="Sentence pacing")
                return fig
            
            lengths = [len(s.split()) for s in sentences]
            fig = go.Figure()
            
            # Plot individual sentence lengths
            fig.add_trace(go.Scatter(
                x=list(range(len(lengths))),
                y=lengths,
                mode='lines+markers',
                line=dict(color='#636efa', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                name='Words'
            ))
            
            # Add average trendline
            avg_words = sum(lengths) / len(lengths) if lengths else 0
            fig.add_hline(
                y=avg_words, line_dash="dash", line_color="orange",
                annotation_text=f"Avg: {avg_words:.1f}", 
                annotation_position="top left"
            )
            
            fig.update_layout(
                title="Sentence Pacing (Words per sentence)",
                xaxis_title="Sentence Index",
                yaxis_title="Word Count",
                height=height,
                template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white',
                margin=dict(t=50, b=50, l=50, r=50),
                showlegend=False
            )
            return fig
        except Exception as e:
            logging.error(f"plot_sentence_flow failed: {e}")
            raise CustomException(e, sys)

    # 3. READABILITY GAUGE
    def plot_readability_gauge(self,
        article_data: Union[pd.Series, pd.DataFrame],
        height: int = 320,
    ) -> go.Figure:
        """Flesch reading ease gauge with interpretation."""
        try:
            data = self._normalize_input(article_data)
            raw_score = data.get('flesch_reading_ease', 0.0)
            score = 0.0 if pd.isna(raw_score) else float(raw_score or 0.0)
            if score <= 30:
                label = "Hard"
            elif score <= 70:
                label = "Standard"
            else:
                label = "Easy"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score, number={"suffix": " — " + label},
                title={"text": "Readability (Flesch)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4563a8"},
                    "steps": [
                        {"range": [0, 30], "color": "#ff4b4b"},
                        {"range": [30, 70], "color": "#ffa500"},
                        {"range": [70, 100], "color": "#00cc96"},
                    ],
                    "threshold": {"line": {"color": "darkblue", "width": 4}, "value": score},
                },
            ))
            fig.update_layout(height=height, margin=dict(t=40, b=30, l=40, r=40), template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white')
            return fig
        except Exception as e:
            logging.error(f"plot_readability_gauge failed: {e}")
            raise CustomException(e, sys)

    # 4. INFORMATION DENSITY: Fact vs. Fluff Ratio
    def plot_information_density(self,
        article_data: Union[pd.Series, pd.DataFrame],
        height: int = 340,
    ) -> go.Figure:
        """Information density profile (ratios)."""
        try:
            data = self._normalize_input(article_data)
            wc = max(int(data.get('word_count', 1)), 1)
            stopword_val = self._safe_get(data, 'stopword_count', 0) or self._safe_get(data, 'stopword_Count', 0)
            
            metrics = {
                'Lexical richness': (data.get('unique_word_count', 0)) / wc,
                'Stopword ratio': stopword_val / wc,
                'Subject (nouns)': (data.get('noun_count', 0)) / wc,
                'Action (verbs)': (data.get('verb_count', 0)) / wc,
                'Fact density (entities)': (data.get('unique_entity_count', 0)) / wc,
            }
            
            df_plot = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            df_plot['Value'] = df_plot['Value'].clip(0, 1)
            
            fig = px.bar(
                df_plot, x='Value', y='Metric', orientation='h',
                title="Information Density (Ratios)", text='Value',
                color='Metric', color_discrete_sequence=px.colors.qualitative.Prism,
                height=height,
            )
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_layout(
                xaxis_tickformat='.0%',
                xaxis_range=[0, 1.1],
                showlegend=False,
                template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white',
                margin=dict(t=50, b=50, l=100, r=50),
                yaxis=dict(automargin=True),
            )
            return fig
        except Exception as e:
            logging.error(f"plot_information_density failed: {e}")
            raise CustomException(e, sys)

    # 5. TEXT STATS OVERVIEW: Compact Metrics
    def plot_text_stats_overview(
        self,
        article_data: Union[pd.Series, pd.DataFrame],
        height: int = 220,
    ) -> go.Figure:
        """Compact overview: word count, sentences, readability, entities."""
        try:
            data = self._normalize_input(article_data)
            wc = int(data.get('word_count', 0) or 0)
            sc = int(data.get('sentence_count', 0) or 0)
            
            raw_flesch = data.get('flesch_reading_ease', 0.0)
            fleisch = 0.0 if pd.isna(raw_flesch) else float(raw_flesch or 0.0)
            
            entities = int(data.get('unique_entity_count', 0) or 0)
            
            fig = make_subplots(
                rows=1, cols=4,
                subplot_titles=('Words', 'Sentences', 'Readability', 'Entities'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
            )
            fig.add_trace(go.Indicator(mode="number", value=wc, title={"text": "Words"}), row=1, col=1)
            fig.add_trace(go.Indicator(mode="number", value=sc, title={"text": "Sentences"}), row=1, col=2)
            fig.add_trace(go.Indicator(mode="number", value=round(fleisch, 1), title={"text": "Flesch"}), row=1, col=3)
            fig.add_trace(go.Indicator(mode="number", value=entities, title={"text": "Entities"}), row=1, col=4)
            fig.update_layout(height=height, template='plotly_white', margin=dict(t=60, b=30))
            
            return fig
        except Exception as e:
            logging.error(f"plot_text_stats_overview failed: {e}")
            raise CustomException(e, sys)

    # 6. DATA TABLE: Interactive Feature View
    def get_features_dataframe(
        self,
        article_data: Union[pd.Series, pd.DataFrame],
        exclude_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a clean DataFrame of features for table display."""
        try:
            data = self._normalize_input(article_data).to_frame().T
            exclude = exclude_columns or [
                'Content', 'Summary', 'embedding', 'Embedding', 'text', 'Topic', 'cluster', 'Cluster',
            ]
            drop = [c for c in exclude if c in data.columns]
            df_view = data.drop(columns=drop, errors='ignore')
            df_melted = df_view.T.reset_index()
            df_melted.columns = ["Feature", "Value"]
            df_melted["Value"] = df_melted["Value"].astype(str)
            return df_melted
        except Exception as e:
            logging.error(f"get_features_dataframe failed: {e}")
            raise CustomException(e, sys)

    ## Additional data visualization
    def get_interactive_data_view(
        self,
        article_data: Union[pd.Series, pd.DataFrame],
    ) -> Union[Any, pd.DataFrame]:
        """Return AgGrid component for interactive feature table."""
        try:
            df_melted = self.get_features_dataframe(article_data)
            gb = GridOptionsBuilder.from_dataframe(df_melted)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            grid_options = gb.build()
            return AgGrid(
                df_melted,
                gridOptions=grid_options,
                data_return_mode=DataReturnMode.AS_INPUT,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                theme='balham',
                enable_enterprise_modules=False,
                height=400,
            )
        except Exception as e:
            logging.error(f"get_interactive_data_view failed: {e}")
            raise CustomException(e, sys)


# ============================================================================
# 2) KMeansVisualizer — Cluster labels & similar articles
# ============================================================================
class KMeansVisualizer:
    """
    KMeans + TF-IDF cluster visualization. Loads KMeansTopicLabeler; uses article
    embedding and content to get cluster and keywords. Returns plots and similar-articles table.
    """
    def __init__(self, article_data: Union[pd.Series, pd.DataFrame], n_clusters: int = 10):
        logging.info("Initializing KMeans Visualizer...")
        self.labeler = KMeansTopicLabeler(n_clusters=n_clusters)
        self.labeler.load()
        self.article_data = self._normalize_input(article_data)
        emb = self.article_data.get('embedding')
        text = self.article_data.get('Content', '')
        self.result = self.labeler.predict(text=text, embedding=emb)

    def _normalize_input(self, article_data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        if isinstance(article_data, pd.DataFrame):
            return article_data.iloc[0]
        return article_data

    # 1. CLUSTER KEYWORDS: TF-IDF Topic Labels
    def plot_cluster_keywords(self, height: int = 360) -> Optional[go.Figure]:
        """Horizontal bar of cluster TF-IDF keywords (importance order)."""
        try:
            keywords = self.result.get('labels') or []
            if not keywords:
                logging.info("No cluster keywords available.")
                return None
            rank = list(range(len(keywords), 0, -1))
            df = pd.DataFrame({'Keyword': keywords, 'Rank': rank})
            fig = px.bar(
                df, x='Rank', y='Keyword', orientation='h', text='Keyword',
                title=f"Cluster {self.result.get('cluster', '?')} — Topic Keywords",
                labels={'Rank': 'Importance', 'Keyword': 'Term'},
                color='Keyword', color_discrete_sequence=px.colors.qualitative.Pastel1, # Softer, stylish colors
                height=height,
            )
            fig.update_traces(textposition='inside', textfont=dict(color='black', size=14),
                              marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(yaxis=dict(autorange='reversed'), showlegend=False, 
                              template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white', 
                              xaxis_visible=False)
            fig.update_layout(margin=dict(t=50, b=50, l=10, r=50), yaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_cluster_keywords failed: {e}")
            raise CustomException(e, sys)

    # 2. CLUSTER FIT GAUGE: Distance to Centroid
    def plot_cluster_fit_gauge(self, height: int = 260) -> go.Figure:
        """Distance of article embedding to cluster centroid (lower = better fit)."""
        try:
            emb = np.asarray(self.article_data['embedding'], dtype=np.float64)
            cid = int(self.result['cluster'])
            centroid = self.labeler.Kmeans.cluster_centers_[cid]
            dist = float(np.linalg.norm(emb - centroid))
            max_dist = 2.0
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dist,
                title={"text": "Distance to cluster centroid<br><span style='font-size:0.8em;color:gray'>(lower = better fit)</span>"},
                gauge={"axis": {"range": [0, max_dist]}, "bar": {"color": "#1f77b4"},
                       "steps": [
                           {"range": [0, 0.5], "color": "rgba(0, 255, 0, 0.1)"},
                           {"range": [0.5, 1.0], "color": "rgba(255, 255, 0, 0.1)"},
                           {"range": [1.0, 2.0], "color": "rgba(255, 0, 0, 0.1)"},
                       ]},
            ))
            fig.update_layout(height=height, template='plotly_white', margin=dict(t=40, b=30))
            return fig
        except Exception as e:
            logging.error(f"plot_cluster_fit_gauge failed: {e}")
            raise CustomException(e, sys)

    def get_similar_articles(
        self,
        full_dataset_path: str = FINAL_DATASET_PARQUET,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Top-k similar articles in the same cluster (cosine similarity)."""
        try:
            if not os.path.exists(full_dataset_path):
                return pd.DataFrame()
                
            from sklearn.metrics.pairwise import cosine_similarity
            df = pd.read_parquet(full_dataset_path)
            cluster_id = str(self.result['cluster'])
            df = df[df['cluster'].astype(str) == cluster_id].copy()
            df = df[df['embedding'].notna()]
            if df.empty:
                return pd.DataFrame()
            emb_matrix = np.vstack(df['embedding'].values)
            query_emb = np.asarray(self.article_data['embedding']).reshape(1, -1)
            sims = cosine_similarity(query_emb, emb_matrix)[0]
            df = df.copy()
            df['similarity'] = sims
            df = df.sort_values('similarity', ascending=False)
            df = df[df['similarity'] < 0.9999].head(top_k)
            cols = [c for c in ['Topic', 'word_count', 'cluster', 'similarity', 'Content', 'Summary'] if c in df.columns]
            return df[cols] if cols else df
        except Exception as e:
            logging.error(f"get_similar_articles failed: {e}")
            raise CustomException(e, sys)

    def cluster_strength(self) -> float:
        """Scalar: distance to centroid."""
        emb = np.asarray(self.article_data['embedding'])
        cid = int(self.result['cluster'])
        centroid = self.labeler.Kmeans.cluster_centers_[cid]
        return float(np.linalg.norm(emb - centroid))


# ============================================================================
# 3) NERVisualizer — Entities & keywords (spaCy, GLiNER, KeyBERT)
# ============================================================================
class NERVisualizer:
    """
    Visualization for NER + keyword extraction. Accepts either:
    - extraction_result: dict from InformationExtractor.process_articles(), or
    - (extractor, article) to run extraction inside the visualizer.
    """
    def __init__(
        self,
        extraction_result: Optional[Dict[str, Any]] = None,
        extractor: Optional[Any] = None,
        article: Optional[str] = None,
    ):
        self.result = extraction_result
        if extraction_result is None and extractor is not None and article:
            logging.info("Running NER + keyword extraction...")
            self.result = extractor.process_articles(article, custom_labels=None)
        if self.result is None:
            self.result = {"Keywords": [], "predefined_ent": ([], None), "custom_ent": []}

    def _spacy_entities_list(self) -> List[Tuple[str, str]]:
        """Match NER.process_articles: predefined_ent is entities list from extract_spacy_entities."""
        pred = self.result.get("predefined_ent")
        
        if pred is None:
            return []
        
        # Handle the old tuple format gracefully if cached
        if isinstance(pred, (list, tuple)) and len(pred) >= 2 and isinstance(pred[1], type(None)):
            entities = pred[0]
        else:
            entities = pred
            
        if not isinstance(entities, (list, tuple)):
            return []
        
        result = []
        for e in entities:
             if isinstance(e, (list, tuple)) and len(e) >= 2:
                 result.append((str(e[0]), str(e[1])))
        return result

    def _gliner_entities_list(self) -> List[Dict[str, Any]]:
        """Match NER.process_articles: custom_ent from extract_gliner_entities (dict with text, label, score/confidence)."""
        custom = self.result.get("custom_ent")
        if custom is None or not isinstance(custom, (list, tuple)):
            return []
        
        out = []
        for e in custom:
            if isinstance(e, dict):
                score = e.get("score") or e.get("confidence") or 0
                out.append({"text": e.get("text", ""), "label": e.get("label", ""), "score": float(score)})
            elif isinstance(e, (list, tuple)):
                out.append({
                    "text": str(e[0]) if len(e) > 0 else "",
                    "label": str(e[1]) if len(e) > 1 else "",
                    "score": float(e[2]) if len(e) > 2 else 0.0,
                })
            else:
                continue
        return out

    # 1. ENTITY DISTRIBUTION: spaCy Named Entities by Type
    def plot_entity_distribution_spacy(self, height: int = 340) -> Optional[go.Figure]:
        """Bar chart: entity type counts (spaCy)."""
        try:
            entities = self._spacy_entities_list()
            if not entities:
                logging.info("No spaCy entities to display.")
                return None
            from collections import Counter
            counts = Counter(e[1] for e in entities)
            df = pd.DataFrame(list(counts.items()), columns=['Type', 'Count']).sort_values('Count', ascending=True)
            fig = px.bar(
                df, x='Count', y='Type', orientation='h', text='Count',
                title="Named Entities by Type (spaCy)",
                color='Type', color_discrete_sequence=px.colors.qualitative.Prism,
                height=height,
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white', margin=dict(t=50, b=50, l=80, r=50), yaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_entity_distribution_spacy failed: {e}")
            raise CustomException(e, sys)

    # 2. ENTITY DISTRIBUTION: GLiNER Zero-Shot Entities by Type
    def plot_entity_distribution_gliner(self, height: int = 340) -> Optional[go.Figure]:
        """Bar chart: entity type counts (GLiNER)."""
        try:
            entities = self._gliner_entities_list()
            if not entities:
                logging.info("No GLiNER entities to display.")
                return None
            from collections import Counter
            counts = Counter(e.get("label") or "?" for e in entities)
            df = pd.DataFrame(list(counts.items()), columns=['Type', 'Count']).sort_values('Count', ascending=True)
            fig = px.bar(
                df, x='Count', y='Type', orientation='h', text='Count',
                title="Named Entities by Type (GLiNER)",
                color='Type', color_discrete_sequence=px.colors.qualitative.Prism,
                height=height,
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white', margin=dict(t=50, b=50, l=80, r=50), yaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_entity_distribution_gliner failed: {e}")
            raise CustomException(e, sys)

    # 3. SEMANTIC KEYWORDS: KeyBERT Extraction
    def plot_keywords(self, height: int = 360) -> Optional[go.Figure]:
        """Horizontal bar: KeyBERT keywords with scores. Keywords from NER are list of (word, score) tuples."""
        try:
            keywords = self.result.get("Keywords")
            if keywords is None or not isinstance(keywords, (list, tuple)) or len(keywords) == 0:
                logging.info("No keywords to display.")
                return None
            words, scores = [], []
            for k in keywords:
                if isinstance(k, (list, tuple)) and len(k) >= 2:
                    words.append(str(k[0]))
                    scores.append(float(k[1]))
                elif isinstance(k, (list, tuple)) and len(k) == 1:
                    words.append(str(k[0]))
                    scores.append(1.0)
                else:
                    words.append(str(k))
                    scores.append(1.0)
            if not words:
                return None
            df = pd.DataFrame({"Keyword": words, "Score": scores})
            fig = px.bar(
                df, x='Score', y='Keyword', orientation='h',
                title="Semantic keywords (KeyBERT)",
                color='Keyword', color_discrete_sequence=px.colors.qualitative.Prism,
                height=height,
            )
            fig.update_layout(yaxis=dict(autorange='reversed'), showlegend=False, template='plotly_white')
            fig.update_layout(margin=dict(t=50, b=50, l=80, r=50), yaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_keywords failed: {e}")
            raise CustomException(e, sys)

    def get_entities_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Tables for frontend: spaCy entities, GLiNER entities, keywords."""
        try:
            spacy_ents = self._spacy_entities_list()
            gliner_ents = self._gliner_entities_list()
            keywords = self.result.get("Keywords") or []
            if keywords and isinstance(keywords[0], (list, tuple)):
                kw_df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
            else:
                kw_df = pd.DataFrame({"Keyword": keywords})
            return {
                "spacy_entities": pd.DataFrame(spacy_ents, columns=["Entity", "Type"]) if spacy_ents else pd.DataFrame(columns=["Entity", "Type"]),
                "gliner_entities": pd.DataFrame(gliner_ents) if gliner_ents else pd.DataFrame(columns=["text", "label", "score"]),
                "keywords": kw_df,
            }
        except Exception as e:
            logging.error(f"get_entities_dataframe failed: {e}")
            raise CustomException(e, sys)


# ============================================================================
# 4) BERTopicVisualizer — Topic assignment & topic keywords
# ============================================================================
class BERTopicVisualizer:
    """
    BERTopic inference visualization. Loads BERTopic from config; runs transform
    on article text to get topic and probabilities.
    Plots topic distribution and topic keywords. using prebuilt codes
    """
    def __init__(
        self,
        article_text: str,
        topic_model=None,
        embedding: Optional[np.ndarray] = None,
    ):
        """
        Either pass precomputed (topic_id, probs) or article_text + optional topic_model.
        If topic_model is None, loads from BERTOPIC_MODEL_PARAMETERS.
        """
        self.article_text = (article_text or "").strip()
        self.topic_id = None
        self.probs = None
        self.topic_model = topic_model

        if topic_model is None and os.path.exists(BERTOPIC_MODEL_PARAMETERS):
            from bertopic import BERTopic
            logging.info("Loading BERTopic for visualization...")
            self.topic_model = BERTopic.load(BERTOPIC_MODEL_PARAMETERS, embedding_model=SENTENCE_TRANSFORMER_MODEL)

        if self.topic_model is not None and self.article_text:
            try:
                topics, probs = self.topic_model.transform([self.article_text])
                self.topic_id = topics[0] if topics is not None else -1
                self.probs = probs[0] if probs is not None else None
            except Exception as e:
                logging.error(f"BERTopic transform failed: {e}")
                self.topic_id = -1
                self.probs = None

    def get_topic_keywords(self, topic_id: Optional[int] = None) -> List[Tuple[str, float]]:
        """Top words for topic from BERTopic.get_topic()."""
        if self.topic_model is None:
            return []
        tid = topic_id if topic_id is not None else self.topic_id
        if tid is None or tid < 0:
            return []
        topic_tuples = self.topic_model.get_topic(tid)
        if not topic_tuples:
            return []
        return [(str(w), float(s)) for w, s in (topic_tuples or [])]

    # 1. TOPIC DISTRIBUTION: BERTopic Probability Scores
    def plot_topic_distribution(self, top_n: int = 10, height: int = 340) -> Optional[go.Figure]:
        """Bar chart of topic probabilities (if available)."""
        try:
            if self.probs is None or not hasattr(self.probs, '__len__'):
                logging.info("No topic probabilities available.")
                return None
            probs = np.asarray(self.probs)
            if probs.size == 0:
                return None
            idx = np.argsort(probs)[::-1][:top_n]
            labels = [f"Topic {int(i)}" for i in idx]
            values = probs[idx].tolist()
            df = pd.DataFrame({'Topic': labels, 'Probability': values})
            fig = px.bar(
                df, x='Topic', y='Probability',
                title="Topic distribution (BERTopic)",
                labels={'Topic': 'Topic', 'Probability': 'Prob'},
                color='Topic', color_discrete_sequence=px.colors.qualitative.Prism,
                height=height,
            )
            fig.update_layout(showlegend=False, template='plotly_white', margin=dict(t=50, b=50, l=50, r=50), xaxis=dict(tickangle=-45, automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_topic_distribution failed: {e}")
            raise CustomException(e, sys)

    # 2. TOPIC KEYWORDS: Representative Terms
    def plot_topic_keywords(self, topic_id: Optional[int] = None, height: int = 360) -> Optional[go.Figure]:
        """Bar chart of top keywords for the assigned topic."""
        try:
            kw = self.get_topic_keywords(topic_id)[:15]
            if not kw:
                logging.info("No topic keywords available.")
                return None
            words, scores = zip(*kw)
            tid = topic_id if topic_id is not None else self.topic_id
            df = pd.DataFrame({'Term': list(words), 'Score': list(scores)})
            fig = px.bar(
                df, x='Score', y='Term', orientation='h',
                title=f"Topic {tid} — representative terms",
                labels={'Score': 'Score', 'Term': 'Term'},
                color='Term', color_discrete_sequence=px.colors.qualitative.Prism,
                height=height,
            )
            fig.update_layout(yaxis=dict(autorange='reversed'), showlegend=False, template='plotly_white')
            fig.update_layout(margin=dict(t=50, b=50, l=80, r=50), yaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_topic_keywords failed: {e}")
            raise CustomException(e, sys)

    def get_topic_info(self) -> Dict[str, Any]:
        """For frontend: topic id, top keywords."""
        kw = self.get_topic_keywords(self.topic_id)
        return {
            "topic_id": self.topic_id,
            "topic_keywords": kw,
        }

    # 3. TOPIC INFO TABLE: BERTopic get_topic_info (manual inspection)
    def get_topic_info_table(self) -> Optional[pd.DataFrame]:
        """Returns BERTopic topic info table (Count, Name, Representation, etc.)."""
        try:
            if self.topic_model is None:
                return None
            return self.topic_model.get_topic_info()
        except Exception as e:
            logging.error(f"get_topic_info_table failed: {e}")
            return None


# ============================================================================
# 5) SummarizationVisualizer — Original vs summary
# ============================================================================
class SummarizationVisualizer:
    """
    Visualizer for BERTopic Clustering Results.
    Integrates the pre-trained BERTopic model to analyze the specific topic assigned to the article.
    Also,
    Compares original article and summary: length, compression ratio, sentence count.
    No model load; only needs original_text and summary_text strings.
    """
    def __init__(self, original_text: str, summary_text: str):
        self.original = (original_text or "").strip()
        self.summary = (summary_text or "").strip()
        self.orig_words = len(self.original.split())
        self.summ_words = len(self.summary.split())
        self.orig_sents = len(_sentences_from_text(self.original))
        self.summ_sents = len(_sentences_from_text(self.summary))
        self.compression = (1 - self.summ_words / max(1, self.orig_words)) * 100

    # 1. LENGTH COMPARISON: Original vs Summary
    def plot_length_comparison(self, height: int = 320) -> go.Figure:
        """Bar chart: word count and sentence count — original vs summary."""
        try:
            df = pd.DataFrame({
                "Metric": ["Words", "Words", "Sentences", "Sentences"],
                "Source": ["Original", "Summary", "Original", "Summary"],
                "Value": [self.orig_words, self.summ_words, self.orig_sents, self.summ_sents],
            })
            fig = px.bar(
                df, x='Metric', y='Value', color='Source', barmode='group',
                title="Original vs summary length",
                color_discrete_sequence=px.colors.qualitative.Prism,
                height=height,
            )
            fig.update_layout(template='plotly_white', margin=dict(t=50, b=50, l=50, r=50), xaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_length_comparison failed: {e}")
            raise CustomException(e, sys)

    # 2. COMPRESSION GAUGE: Summary Length
    def plot_compression_gauge(self, height: int = 280) -> go.Figure:
        """Compression ratio gauge (%)."""
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(self.compression, 1),
                title={"text": "Compression (%)"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00cc96"},
                    "steps": [
                        {"range": [0, 33], "color": "#e8f4f8"},
                        {"range": [33, 66], "color": "#b8e0d2"},
                        {"range": [66, 100], "color": "#00cc96"},
                    ],
                },
            ))
            fig.update_layout(height=height, template='plotly_white', margin=dict(t=40, b=30))
            return fig
        except Exception as e:
            logging.error(f"plot_compression_gauge failed: {e}")
            raise CustomException(e, sys)

    def get_summary_stats(self) -> Dict[str, Any]:
        """For frontend: word/sentence counts and compression."""
        return {
            "original_words": self.orig_words,
            "summary_words": self.summ_words,
            "original_sentences": self.orig_sents,
            "summary_sentences": self.summ_sents,
            "compression_pct": round(self.compression, 2),
        }


# ---------------------------------------------------------------------------
# Demo (Streamlit)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(page_title="Visualizer Test", layout="wide")
    st.title("News Analysis — Inference Visualizations")

    try:
        if not os.path.exists(FINAL_DATASET_PARQUET):
            raise FileNotFoundError(f"Dataset not found: {FINAL_DATASET_PARQUET}")
        df = pd.read_parquet(FINAL_DATASET_PARQUET)
        row = df.sample(1)

        # NewsVisualizer
        nv = NewsVisualizer()
        st.subheader("1. Heuristic features")
        st.plotly_chart(nv.plot_text_stats_overview(row), width='stretch')
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(nv.plot_grammar_composition(row, 'bar'), width='stretch')
            st.plotly_chart(nv.plot_sentence_flow(row), width='stretch')
        with c2:
            st.plotly_chart(nv.plot_readability_gauge(row), width='stretch')
            st.plotly_chart(nv.plot_information_density(row), width='stretch')
        st.subheader("Feature table")
        st.dataframe(nv.get_features_dataframe(row), width='stretch')

        # KMeansVisualizer
        try:
            kv = KMeansVisualizer(row)
            st.subheader("2. KMeans cluster")
            kw_fig = kv.plot_cluster_keywords()
            if kw_fig is not None:
                st.plotly_chart(kw_fig, width='stretch')
            st.plotly_chart(kv.plot_cluster_fit_gauge(), width='stretch')
            st.metric("Cluster distance", round(kv.cluster_strength(), 4))
            similar_df = kv.get_similar_articles(top_k=5)
            if not similar_df.empty:
                st.dataframe(similar_df, width='stretch')
        except Exception as kv_err:
            logging.warning(f"KMeans visualization skipped: {kv_err}")
            st.warning(f"KMeans cluster view unavailable: {kv_err}")

        # BERTopic
        if os.path.exists(BERTOPIC_MODEL_PARAMETERS):
            try:
                bv = BERTopicVisualizer(row.iloc[0].get('Content', ''))
                if bv.topic_model:
                    st.subheader("3. BERTopic")
                    # Inference: article's topic distribution and keywords
                    dist_fig = bv.plot_topic_distribution()
                    if dist_fig is not None:
                        st.plotly_chart(dist_fig, width='stretch')
                    kw_fig = bv.plot_topic_keywords()
                    if kw_fig is not None:
                        st.plotly_chart(kw_fig, width='stretch')
                    # Topic info table (manual inspection)
                    topic_info_df = bv.get_topic_info_table()
                    if topic_info_df is not None and not topic_info_df.empty:
                        st.caption("Topic information (manual inspection)")
                        st.dataframe(topic_info_df, width='stretch', height=300)
                    # Aggregation table by topic
                    agg_df = get_bertopic_aggregation_table(df, topic_col="Topic")
                    if agg_df is not None and not agg_df.empty:
                        st.caption("Aggregation by topic (mean of features)")
                        st.dataframe(agg_df, width='stretch', height=250)
                    # Built-in BERTopic plots
                    st.caption("Dataset-level: topic relationships and hierarchy")
                    viz_topics = bv.plot_visualize_topics()
                    if viz_topics is not None:
                        st.plotly_chart(viz_topics, width='stretch')
                    viz_heat = bv.plot_visualize_heatmap()
                    if viz_heat is not None:
                        st.plotly_chart(viz_heat, width='stretch')
                    viz_hier = bv.plot_visualize_hierarchy()
                    if viz_hier is not None:
                        st.plotly_chart(viz_hier, width='stretch')
            except Exception as bv_err:
                logging.warning(f"BERTopic visualization skipped: {bv_err}")
                st.warning(f"BERTopic view unavailable: {bv_err}")

        # Summarization
        if 'Summary' in row.columns and pd.notna(row.iloc[0].get('Summary')):
            try:
                sv = SummarizationVisualizer(row.iloc[0].get('Content', ''), row.iloc[0].get('Summary', ''))
                st.subheader("4. Summarization")
                st.plotly_chart(sv.plot_length_comparison(), width='stretch')
                st.plotly_chart(sv.plot_compression_gauge(), width='stretch')
            except Exception as sv_err:
                logging.warning(f"Summarization visualization skipped: {sv_err}")
                st.warning(f"Summarization view unavailable: {sv_err}")

        # NER
        st.subheader("5. NER & keywords")
        try:
            from src.Components.NER import InformationExtractor
            ext = InformationExtractor()
            nerr = ext.process_articles(row.iloc[0].get('Content', ''))
            nv_ner = NERVisualizer(extraction_result=nerr)
            spacy_fig = nv_ner.plot_entity_distribution_spacy()
            gliner_fig = nv_ner.plot_entity_distribution_gliner()
            kw_fig = nv_ner.plot_keywords()
            if spacy_fig is None and gliner_fig is None and kw_fig is None:
                st.info("No entities or keywords extracted for this article.")
            else:
                ner_col1, ner_col2 = st.columns(2)
                with ner_col1:
                    if spacy_fig is not None:
                        st.plotly_chart(spacy_fig, width='stretch')
                    if gliner_fig is not None:
                        st.plotly_chart(gliner_fig, width='stretch')
                with ner_col2:
                    if kw_fig is not None:
                        st.plotly_chart(kw_fig, width='stretch')
        except Exception as ner_err:
            logging.warning(f"NER visualization skipped: {ner_err}")
            st.warning(f"NER view unavailable (models may not be loaded): {ner_err}")

        st.success("Visualization test completed.")
    except Exception as e:
        logging.error(f"Visualization test failed: {e}")
        st.error(str(e))
