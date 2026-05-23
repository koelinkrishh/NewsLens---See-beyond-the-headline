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

from src.logger import logging
from src.exception import CustomException

# 1. Suppress the oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _sentences_from_text(text: str) -> List[str]:
    """Split text into sentences (regex-based)."""
    if not text or not str(text).strip():
        return []
    parts = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [p.strip() for p in parts if len(p.split()) > 0]


class ArticleVisualizer:
    """
    Unified visualization engine for the updated LC-based backend components.
    Provides standalone methods to generate Plotly figures for Features, NER, 
    Summaries, and Recommendations to be seamlessly rendered in Streamlit.
    """
    def __init__(self):
        logging.info("Initializing Unified ArticleVisualizer...")
        self.colors = px.colors.qualitative.Prism

    # ============================================================================
    # 1) Feature Generation Visuals (from FeatureGeneration_lc output)
    # ============================================================================
    
    def plot_grammar_composition(self, features: dict, height: int = 380) -> go.Figure:
        """POS composition pie chart based on extracted features."""
        try:
            pos_keys = ['noun_count', 'verb_count', 'adj_count', 'adv_count', 'pronoun_count']
            pos_labels = ['Nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Pronouns']
            
            pos_counts = [max(0, int(features.get(k, 0))) for k in pos_keys]
            df_plot = pd.DataFrame({'POS': pos_labels, 'Count': pos_counts})
            df_plot = df_plot[df_plot['Count'] > 0]
            
            if df_plot.empty:
                df_plot = pd.DataFrame({'POS': pos_labels, 'Count': [0] * len(pos_labels)})

            fig = px.pie(
                df_plot, values='Count', names='POS',
                title="Grammatical Composition",
                hole=0.5, height=height, color_discrete_sequence=self.colors,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
            fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
            return fig
        except Exception as e:
            logging.error(f"plot_grammar_composition failed: {e}")
            raise CustomException(e, sys)

    def plot_sentence_flow(self, text: str, height: int = 340) -> go.Figure:
        """Sentence pacing: word count per sentence."""
        try:
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
                margin=dict(t=50, b=50, l=50, r=50),
                showlegend=False
            )
            return fig
        except Exception as e:
            logging.error(f"plot_sentence_flow failed: {e}")
            raise CustomException(e, sys)

    def plot_readability_gauge(self, flesch_score: float, height: int = 320) -> go.Figure:
        """Flesch reading ease gauge with interpretation."""
        try:
            score = 0.0 if pd.isna(flesch_score) else float(flesch_score)
            
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
            fig.update_layout(height=height, margin=dict(t=40, b=30, l=40, r=40))
            return fig
        except Exception as e:
            logging.error(f"plot_readability_gauge failed: {e}")
            raise CustomException(e, sys)


    # ============================================================================
    # 2) NER & Keyword Visuals (from NER_lc output)
    # ============================================================================

    def plot_entity_distribution(self, entities: List[dict], title: str = "Named Entities by Type", height: int = 340) -> Optional[go.Figure]:
        """Bar chart: entity type counts. Works for both predefined and custom entities."""
        try:
            if not entities:
                return None
            
            from collections import Counter
            counts = Counter(e.get("label", "?") for e in entities)
            df = pd.DataFrame(list(counts.items()), columns=['Type', 'Count']).sort_values('Count', ascending=True)
            
            fig = px.bar(
                df, x='Count', y='Type', orientation='h', text='Count',
                title=title,
                color='Type', color_discrete_sequence=self.colors,
                height=height,
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, margin=dict(t=50, b=50, l=80, r=50), yaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_entity_distribution failed: {e}")
            raise CustomException(e, sys)

    def plot_keywords(self, keywords: List[Tuple[str, float]], height: int = 360) -> Optional[go.Figure]:
        """Horizontal bar: KeyBERT keywords with scores."""
        try:
            if not keywords:
                return None
            
            words, scores = [], []
            for k in keywords:
                if isinstance(k, (list, tuple)) and len(k) >= 2:
                    words.append(str(k[0]))
                    scores.append(float(k[1]))
            
            df = pd.DataFrame({"Keyword": words, "Score": scores})
            fig = px.bar(
                df, x='Score', y='Keyword', orientation='h',
                title="Semantic keywords (KeyBERT)",
                color='Keyword', color_discrete_sequence=self.colors,
                height=height,
            )
            fig.update_layout(yaxis=dict(autorange='reversed'), showlegend=False)
            fig.update_layout(margin=dict(t=50, b=50, l=80, r=50), yaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_keywords failed: {e}")
            raise CustomException(e, sys)


    # ============================================================================
    # 3) Summarization Visuals (from Summarization_lc output)
    # ============================================================================

    def plot_summary_comparison(self, original_text: str, summary_text: str, height: int = 320) -> go.Figure:
        """Bar chart: word count and sentence count — original vs summary."""
        try:
            orig_words = len(original_text.split())
            summ_words = len(summary_text.split())
            orig_sents = len(_sentences_from_text(original_text))
            summ_sents = len(_sentences_from_text(summary_text))

            df = pd.DataFrame({
                "Metric": ["Words", "Words", "Sentences", "Sentences"],
                "Source": ["Original", "Summary", "Original", "Summary"],
                "Value": [orig_words, summ_words, orig_sents, summ_sents],
            })
            
            fig = px.bar(
                df, x='Metric', y='Value', color='Source', barmode='group',
                title="Original vs Summary Length",
                color_discrete_sequence=self.colors,
                height=height,
            )
            fig.update_layout(margin=dict(t=50, b=50, l=50, r=50), xaxis=dict(automargin=True))
            return fig
        except Exception as e:
            logging.error(f"plot_summary_comparison failed: {e}")
            raise CustomException(e, sys)

    def plot_compression_gauge(self, original_text: str, summary_text: str, height: int = 280) -> go.Figure:
        """Compression ratio gauge (%)."""
        try:
            orig_words = max(1, len(original_text.split()))
            summ_words = len(summary_text.split())
            compression = (1 - (summ_words / orig_words)) * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(compression, 1),
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
            fig.update_layout(height=height, margin=dict(t=40, b=30))
            return fig
        except Exception as e:
            logging.error(f"plot_compression_gauge failed: {e}")
            raise CustomException(e, sys)


    # ============================================================================
    # 4) Recommendation Visuals (from Recommendation output)
    # ============================================================================

    def plot_recommendations(self, df_recs: pd.DataFrame, height: int = 400) -> Optional[go.Figure]:
        """Horizontal bar chart showing Recommended Articles and their Distance/Similarity score."""
        try:
            if df_recs is None or df_recs.empty:
                return None
                
            # Assume df has 'id', 'distance', and 'summary_text' or similar columns.
            # Convert distances to similarity if needed, or just plot distance directly.
            if 'distance' in df_recs.columns:
                score_col = 'distance'
                title = "Recommended Articles (L2 Distance - Lower is closer)"
            elif 'similarity' in df_recs.columns:
                score_col = 'similarity'
                title = "Recommended Articles (Cosine Similarity - Higher is closer)"
            else:
                return None
                
            # Create a clean label column (truncate article text for display)
            if 'summary_text' in df_recs.columns:
                df_recs['Display'] = df_recs['summary_text'].apply(lambda x: str(x)[:45] + "..." if len(str(x)) > 45 else str(x))
            elif 'id' in df_recs.columns:
                df_recs['Display'] = "Article ID: " + df_recs['id'].astype(str)
            else:
                df_recs['Display'] = "Article " + df_recs.index.astype(str)

            # Sort ascending if distance (we want smallest distance at top), descending if similarity
            ascending_sort = True if score_col == 'distance' else False
            df_recs = df_recs.sort_values(by=score_col, ascending=ascending_sort).head(10)

            fig = px.bar(
                df_recs, x=score_col, y='Display', orientation='h',
                title=title,
                text=score_col,
                color=score_col, color_continuous_scale="Viridis",
                height=height,
            )
            
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            
            # If distance, reverse y-axis so smallest distance is at top
            if score_col == 'distance':
                fig.update_layout(yaxis=dict(autorange='reversed'))
                
            fig.update_layout(margin=dict(t=50, b=50, l=10, r=50), yaxis=dict(automargin=True))
            return fig
            
        except Exception as e:
            logging.error(f"plot_recommendations failed: {e}")
            raise CustomException(e, sys)


# ---------------------------------------------------------------------------
# Simple Demo & Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing ArticleVisualizer Setup...")
    try:
        vis = ArticleVisualizer()
        
        # Test Feature Visuals
        sample_features = {'noun_count': 50, 'verb_count': 20, 'adj_count': 15, 'adv_count': 5, 'pronoun_count': 10}
        fig1 = vis.plot_grammar_composition(sample_features)
        
        # Test Recommendation Visuals
        sample_recs = pd.DataFrame({
            'id': [1, 2, 3],
            'summary_text': ["Stock markets rise today", "Tech stocks rally heavily", "Inflation drops this quarter"],
            'distance': [0.45, 0.61, 0.82]
        })
        fig2 = vis.plot_recommendations(sample_recs)
        
        # Test Summary Comparison
        orig = "This is a long sentence. It has many words. Let us see."
        summ = "This is short."
        fig3 = vis.plot_summary_comparison(orig, summ)
        
        print("[SUCCESS] ArticleVisualizer instantiated and test plots generated without errors.")
    except Exception as ex:
        print(f"Error during Visualization_lc tests: {ex}")
