"""
app_lc.py — Streamlit Frontend for NewLens LC
Talks to api_lc.py and renders ArticleVisualizer plots.
"""
import pandas as pd
import numpy as np
import requests
import os
import sys
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import API_KEY, MAX_TEXT_LENGTH, API_PORT
from src.Components.Visualization_lc import ArticleVisualizer

# Suppress TF noise
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_URL", f"http://127.0.0.1:{API_PORT}")

st.set_page_config(
    page_title="NewLens — News Intelligence Platform",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
for key, default in {
    "article_text": "",
    "word_count": 0,
    "features_df": None,
    "article_embedding": None,
    "api_online": False,
    "summary_text": None,
    "ner_data": None,
    "qa_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
viz = ArticleVisualizer()

def check_api():
    try:
        r = requests.get(API_BASE_URL + "/", timeout=3)
        return r.status_code in (200, 404)
    except Exception as e:
        return e

def api_headers(user_key: str) -> dict:
    return {"X-API-Key": user_key, "Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
user_api_key = st.sidebar.text_input("API Key", value=API_KEY, type="password")
st.sidebar.divider()
st.sidebar.caption(f"Backend: `{API_BASE_URL}`")

# ---------------------------------------------------------------------------
# API Online Check
# ---------------------------------------------------------------------------
if not st.session_state["api_online"]:
    res = check_api()
    if res is True:
        st.session_state["api_online"] = True
        st.rerun()
    else:
        st.error(f"❌ Backend API is offline at **{API_BASE_URL}**")
        st.warning(f"Connection error: `{res}`")
        st.info("Run in a separate terminal:\n```\nuvicorn app.api_lc:app --reload --port 8001\n```")
        st.stop()

# ---------------------------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------------------------
st.title("📰 NewLens — News Intelligence Platform")

nav_options = [
    "🏠 Home",
    "📊 Feature Analysis",
    "📝 Summarization",
    "🏷️ Named Entity Recognition",
    "💬 Q&A",
    "🔗 Recommendations",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "🏠 Home"

page = st.radio("Navigate", nav_options, horizontal=True, label_visibility="collapsed", key="nav_page")
st.divider()

# ---------------------------------------------------------------------------
# PAGE: Home — Article Input & Processing
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.header("Upload or Paste Article")

    input_method = st.radio("Input Method", ["Paste Text", "Upload .txt File"], horizontal=True)
    raw_text = ""

    if input_method == "Paste Text":
        raw_text = st.text_area(
            "Article Text",
            value=st.session_state["article_text"],
            height=280,
            placeholder="Paste your news article here...",
            max_chars=MAX_TEXT_LENGTH,
        )
    else:
        uploaded = st.file_uploader("Choose a .txt file", type=["txt"])
        if uploaded:
            raw_text = uploaded.getvalue().decode("utf-8")
            st.text_area("File Content", value=raw_text, height=250, disabled=True)

    if st.button("🚀 Process Article", type="primary"):
        if not raw_text.strip():
            st.warning("Please provide article text first.")
        else:
            with st.spinner("Processing via API..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/process",
                        json={"text": raw_text},
                        headers=api_headers(user_api_key),
                        timeout=60,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["article_text"] = raw_text
                        st.session_state["word_count"] = data.get("word_count", len(raw_text.split()))

                        feat_df = pd.DataFrame([data["features"]])
                        feat_df["Content"] = raw_text
                        st.session_state["features_df"] = feat_df
                        st.session_state["article_embedding"] = np.array(data["embedding"])
                        
                        # Clear cache for new article
                        st.session_state["summary_text"] = None
                        st.session_state["ner_data"] = None
                        st.session_state["qa_history"] = []

                        st.success(f"✅ Article processed! ({st.session_state['word_count']} words)")
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Processing Failed: {e}")

    # Display quick stats if available
    if st.session_state["features_df"] is not None:
        df_f = st.session_state["features_df"]
        st.subheader("Quick Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Characters", int(df_f.get("char_count", [0]).iloc[0]))
        c2.metric("Words", st.session_state["word_count"])
        c3.metric("Sentences", int(df_f.get("sentence_count", [0]).iloc[0]))
        c4.metric("Entities", int(df_f.get("unique_entity_count", [0]).iloc[0]))

        # Semantic features (shown when available)
        if "semantic_tone" in df_f.columns:
            st.divider()
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Tone", str(df_f["semantic_tone"].iloc[0]))
            sc2.metric("Objectivity", f"{df_f['semantic_objectivity_score'].iloc[0]:.2f}")
            sc3.metric("Category", str(df_f["semantic_primary_category"].iloc[0]))


# Guard: all other pages require a processed article
if page != "🏠 Home":
    if not st.session_state["article_text"] or st.session_state["features_df"] is None:
        st.info("ℹ️ Please process an article on the **Home** page first.")
        st.stop()

# ---------------------------------------------------------------------------
# PAGE: Feature Analysis
# ---------------------------------------------------------------------------
if page == "📊 Feature Analysis":
    st.header("Heuristic Feature Analysis")
    feat_df = st.session_state["features_df"]
    feats = feat_df.iloc[0].to_dict()
    article_text = st.session_state["article_text"]

    try:
        c1, c2 = st.columns(2)
        with c1:
            # Grammar Composition Pie
            grammar_fig = viz.plot_grammar_composition(feats)
            if grammar_fig:
                st.plotly_chart(grammar_fig, width='stretch')

            # Readability Gauge
            flesch = float(feats.get("flesch_reading_ease", 0.0) or 0.0)
            st.plotly_chart(viz.plot_readability_gauge(flesch), width='stretch')

        with c2:
            # Sentence Flow
            st.plotly_chart(viz.plot_sentence_flow(article_text), width='stretch')

        st.divider()
        st.subheader("Feature Table")
        display_cols = [k for k in feats if k not in ("Content", "embedding")]
        st.dataframe(
            pd.DataFrame({"Feature": display_cols, "Value": [str(feats[k]) for k in display_cols]}),
            width='stretch',
            height=400,
        )
    except Exception as e:
        st.error(f"Error rendering Feature Analysis: {e}")


# ---------------------------------------------------------------------------
# PAGE: Summarization
# ---------------------------------------------------------------------------
elif page == "📝 Summarization":
    st.header("Hybrid Summarization")
    words = st.session_state["word_count"]

    if words < 20:
        st.warning(f"Article too short ({words} words, minimum 20 required).")
    else:
        reduction = st.slider(
            "Reduction Ratio",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="0.1 = very short summary, 0.9 = almost full length"
        )

        if st.session_state.get("summary_text"):
            summary_text = st.session_state["summary_text"]
            st.subheader("Summary")
            st.write(summary_text.replace(". ", ".\n\n"))

            st.divider()
            st.subheader("Compression Analysis")
            c1, c2 = st.columns(2)
            orig = st.session_state["article_text"]
            with c1:
                st.plotly_chart(viz.plot_summary_comparison(orig, summary_text), width='stretch')
            with c2:
                st.plotly_chart(viz.plot_compression_gauge(orig, summary_text), width='stretch')

            if st.button("🔄 Regenerate Summary"):
                st.session_state["summary_text"] = None
                st.rerun()

        else:
            if st.button("✨ Generate Summary", type="primary"):
                with st.spinner("Summarizing with Groq Llama..."):
                    try:
                        res = requests.post(
                            f"{API_BASE_URL}/summarize",
                            json={"text": st.session_state["article_text"], "reduction_ratio": reduction},
                            headers=api_headers(user_api_key),
                            timeout=120,
                        )
                        if res.status_code == 200:
                            st.session_state["summary_text"] = res.json()["summary"]
                            st.rerun()
                        else:
                            st.error(f"Summarization Error: {res.text}")
                    except Exception as e:
                        st.error(f"Summarization Failed: {e}")


# ---------------------------------------------------------------------------
# PAGE: NER
# ---------------------------------------------------------------------------
elif page == "🏷️ Named Entity Recognition":
    st.header("Named Entity Recognition")

    data = st.session_state.get("ner_data")

    if not data:
        if st.button("🏷️ Extract Entities", type="primary"):
            with st.spinner("Extracting entities via Groq LLM..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/ner",
                        json={"text": st.session_state["article_text"]},
                        headers=api_headers(user_api_key),
                        timeout=120,
                    )
                    if res.status_code == 200:
                        st.session_state["ner_data"] = res.json()
                        st.rerun()
                    else:
                        st.error(f"NER Error: {res.text}")
                except Exception as e:
                    st.error(f"NER Failed: {e}")
    else:
        if st.button("🔄 Regenerate NER"):
            st.session_state["ner_data"] = None
            st.rerun()

        st.markdown(
            """
            <style>
            .vertical-line {
                border-left: 2px solid #ddd;
                height: 100%;
                padding-left: 20px;
                min-height: 500px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        col_left, col_right = st.columns([0.6, 0.4])

        with col_left:
            st.subheader("Highlighted Entities")
            html = data.get("html", "")
            st.components.v1.html(
                f"<div style='background:white;color:black;padding:15px;border-radius:8px;"
                f"border:1px solid #e0e0e0;font-family:-apple-system,BlinkMacSystemFont,"
                f"\"Segoe UI\",sans-serif;line-height:1.8;'>{html}</div>",
                height=450, scrolling=True,
            )

            st.subheader("Entities Table")
            all_ents = data.get("entities", [])
            if all_ents:
                df_ent = pd.DataFrame(all_ents)
                df_ent = df_ent.rename(columns={"text": "Entity", "label": "Type", "score": "Score"})
                df_ent = df_ent[["Entity", "Type", "Score"]].sort_values("Score", ascending=False)
                st.dataframe(
                    df_ent.style.background_gradient(cmap="Greens", subset=["Score"]),
                    width='stretch',
                )
            else:
                st.info("No entities extracted.")

            custom_ents = data.get("custom_ents", [])
            if custom_ents:
                st.divider()
                fig = viz.plot_entity_distribution(custom_ents, title="Entity Distribution (by Type)")
                if fig:
                    st.plotly_chart(fig, width='stretch')

        with col_right:
            st.markdown('<div class="vertical-line">', unsafe_allow_html=True)
            st.subheader("Keywords (KeyBERT)")
            keywords = data.get("keywords", [])
            if keywords:
                df_kw = pd.DataFrame(keywords, columns=["Keyword", "Score"])
                st.dataframe(
                    df_kw.style.background_gradient(cmap="Blues", subset=["Score"]),
                    width='stretch', height=350,
                )
                kw_fig = viz.plot_keywords(keywords)
                if kw_fig:
                    st.plotly_chart(kw_fig, width='stretch')
            else:
                st.info("No keywords extracted.")
            st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Q&A
# ---------------------------------------------------------------------------
elif page == "💬 Q&A":
    st.header("Article Q&A (RAG)")
    st.caption("Ask any question about the article — the answer is grounded in actual retrieved chunks via FAISS.")

    question = st.text_input(
        "Your Question",
        placeholder="e.g. What is the main topic of this article?",
    )

    if st.button("🤔 Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/qa",
                        json={
                            "article": st.session_state["article_text"],
                            "question": question,
                        },
                        headers=api_headers(user_api_key),
                        timeout=120,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.subheader("Answer")
                        st.success(data["answer"])

                        st.divider()
                        st.subheader("📖 Source Context Chunks")
                        chunks = data.get("context_chunks", [])
                        if chunks:
                            for i, chunk in enumerate(chunks, 1):
                                with st.expander(f"Chunk {i}"):
                                    st.write(chunk)
                        else:
                            st.info("No context chunks retrieved.")
                    else:
                        st.error(f"Q&A Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Q&A Failed: {e}")


# ---------------------------------------------------------------------------
# PAGE: Recommendations
# ---------------------------------------------------------------------------
elif page == "🔗 Recommendations":
    st.header("Similar Article Recommendations")
    st.caption("Finds the most semantically similar articles from the PostgreSQL pgvector database.")

    top_k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    if st.button("🔍 Find Similar Articles", type="primary"):
        with st.spinner("Querying pgvector recommendation engine..."):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/recommend",
                    json={"text": st.session_state["article_text"], "top_k": top_k},
                    headers=api_headers(user_api_key),
                    timeout=60,
                )
                if res.status_code == 200:
                    data = res.json()
                    recs = data.get("recommendations", [])

                    if not recs:
                        st.info("No similar articles found. Is the dataset ingested into PostgreSQL?")
                    else:
                        df_recs = pd.DataFrame(recs)
                        st.subheader(f"Top {len(recs)} Similar Articles")

                        # Bar chart
                        rec_fig = viz.plot_recommendations(df_recs.copy())
                        if rec_fig:
                            st.plotly_chart(rec_fig, width='stretch')

                        st.divider()
                        # Article cards
                        for i, row in df_recs.iterrows():
                            with st.container(border=True):
                                sc1, sc2 = st.columns([5, 1])
                                with sc1:
                                    score = row.get("similarity_score", row.get("distance", 0.0))
                                    st.caption(f"Score: {float(score):.4f} | ID: {row.get('id', i)}")
                                    # Show summary or article snippet
                                    content = row.get("summary_text") or row.get("article_text", "")
                                    st.write(str(content)[:300] + "..." if len(str(content)) > 300 else str(content))
                                with sc2:
                                    if st.button("Analyze →", key=f"rec_analyze_{i}"):
                                        article_content = row.get("article_text", "")
                                        if article_content:
                                            with st.spinner("Processing..."):
                                                try:
                                                    r2 = requests.post(
                                                        f"{API_BASE_URL}/process",
                                                        json={"text": article_content},
                                                        headers=api_headers(user_api_key),
                                                        timeout=60,
                                                    )
                                                    if r2.status_code == 200:
                                                        d2 = r2.json()
                                                        st.session_state["article_text"] = article_content
                                                        st.session_state["word_count"] = d2.get("word_count", len(article_content.split()))
                                                        feat_df2 = pd.DataFrame([d2["features"]])
                                                        feat_df2["Content"] = article_content
                                                        st.session_state["features_df"] = feat_df2
                                                        st.session_state["article_embedding"] = np.array(d2["embedding"])
                                                        st.session_state["nav_page"] = "🏠 Home"
                                                        
                                                        # Clear cache for recommended article
                                                        st.session_state["summary_text"] = None
                                                        st.session_state["ner_data"] = None
                                                        st.session_state["qa_history"] = []
                                                        
                                                        st.rerun()
                                                    else:
                                                        st.error(f"Process Error: {r2.text}")
                                                except Exception as ex:
                                                    st.error(f"Failed: {ex}")
                else:
                    if res.status_code == 503:
                        st.warning("Recommender is offline. Check PostgreSQL connection and ensure the dataset is ingested.")
                    else:
                        st.error(f"Recommendation Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Recommendation Failed: {e}")
        
