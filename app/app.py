import pandas as pd
import numpy as np
import plotly.express as px
import requests
import os
import sys
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Suppress the oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 2. Suppress other TensorFlow logging (0=all, 1=no INFO, 2=no INFO/WARN, 3=no ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="News Intelligence Platform", layout="wide")

# --- Local Component Imports (For Visualizations) ---
from src.Components.Visualization import (
    NewsVisualizer, KMeansVisualizer, NERVisualizer,
    BERTopicVisualizer, SummarizationVisualizer
)

# --- Main App State Setup ---
if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""
if "word_count" not in st.session_state:
    st.session_state["word_count"] = 0
if "features_df" not in st.session_state:
    st.session_state["features_df"] = None
if "article_embedding" not in st.session_state:
    st.session_state["article_embedding"] = None
if "api_online" not in st.session_state:
    st.session_state["api_online"] = False

# --- Helper to Check API Status ---
def check_api():
    try:
        response = requests.get(API_BASE_URL + "/", timeout=2)
        return response.status_code == 200 or response.status_code == 404
    except Exception as e:
        return e

# --- Main App Layout ---
st.title("📰 News Intelligence Platform (API Mode)")

if not st.session_state["api_online"]:
    res = check_api()
    if res is True:
        st.session_state["api_online"] = True
        st.rerun()
    else:
        st.error(f"❌ Backend API is offline at {API_BASE_URL}")
        st.warning(f"Connection Error: {res}")
        st.info("Ensure Terminal 1 is running: `uvicorn app.api:app --reload` and wait for 'Startup complete'.")
        st.stop()

# Top Navigation
nav_options = [
    "Home", 
    "1. Heuristic Analysis", 
    "2. Summarization", 
    "3. Clustering & Topic Modeling", 
    "4. Named Entity Recognition"
]
page = st.radio("Navigate Modules", nav_options, horizontal=True)

st.divider()

if page == "Home":
    st.header("Upload or Paste Article")
    
    input_method = st.radio("Input Method", ["Paste Text", "Upload Text File"], horizontal=True)
    raw_text = ""
    if input_method == "Paste Text":
        raw_text = st.text_area("Article Text", height=250, placeholder="Paste your news article here...")
    else:
        uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
        if uploaded_file is not None:
            raw_text = uploaded_file.getvalue().decode('utf-8')
            st.text_area("File Content", value=raw_text, height=250, disabled=True)
            
    if st.button("Process Article", type="primary"):
        with st.spinner("Processing via API..."):
            try:
                # 1. Process via API
                response = requests.post(f"{API_BASE_URL}/process", json={"text": raw_text})
                if response.status_code == 200:
                    data = response.json()
                    
                    st.session_state["article_text"] = raw_text
                    st.session_state["word_count"] = len(raw_text.split())
                    
                    # Store features as DataFrame
                    feat_df = pd.DataFrame([data["features"]])
                    feat_df['Content'] = raw_text
                    st.session_state["features_df"] = feat_df
                    
                    # Store embedding as numpy
                    st.session_state["article_embedding"] = np.array(data["embedding"])
                    
                    st.success(f"Article processed successfully! (Word count: {st.session_state['word_count']})")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Processing Failed: {e}")

    # Display basic stats if available
    if st.session_state["features_df"] is not None:
        st.subheader("Basic Statistics")
        df_f = st.session_state["features_df"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Characters", int(df_f['char_count'].values[0]))
        col2.metric("Sentences", int(df_f['sentence_count'].values[0]))
        col3.metric("Words", int(df_f['word_count'].values[0]))
        col4.metric("Entities", int(df_f['unique_entity_count'].values[0]))

# Only allow module access if text is processed
if page != "Home":
    if not st.session_state["article_text"] or st.session_state["features_df"] is None:
        st.info("Please process an article on the Home page first to access this module.")
        st.stop()


if page == "1. Heuristic Analysis":
    st.header("Heuristic Analysis")
    try:
        feat_df = st.session_state["features_df"]
        viz = NewsVisualizer()
        
        st.subheader("Interactive Feature Data View")
        viz.get_interactive_data_view(feat_df)
        
        st.subheader("Heuristic Plots")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.plot_sentence_flow(feat_df), width='stretch')
            st.plotly_chart(viz.plot_information_density(feat_df), width='stretch')
        with c2:
            st.plotly_chart(viz.plot_grammar_composition(feat_df, plot_type='bar'), width='stretch')
            st.plotly_chart(viz.plot_readability_gauge(feat_df), width='stretch')
            
    except Exception as e:
        st.error(f"Error rendering Heuristic Analysis: {e}")


elif page == "2. Summarization":
    st.header("Summarization")
    words = st.session_state["word_count"]
    
    if words < 20:
        st.warning(f"Your article has only {words} words (minimum 20 required). Summarization is disabled for short text.")
    else:
        compression = st.slider("Compression Ratio", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/summarize", 
                                        json={"text": st.session_state["article_text"], "compression": compression})
                    if res.status_code == 200:
                        summary_text = res.json()["summary"]
                        st.subheader("Summary")
                        st.write(summary_text.replace(". ", ".\n\n"))
                        
                        # Visualization (Running locally on API result)
                        st.subheader("Length Comparison")
                        sviz = SummarizationVisualizer(st.session_state["article_text"], summary_text)
                        c1, c2 = st.columns(2)
                        with c1: st.plotly_chart(sviz.plot_length_comparison(), width='stretch')
                        with c2: st.plotly_chart(sviz.plot_compression_gauge(), width='stretch')
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Summarization Failed: {e}")


elif page == "3. Clustering & Topic Modeling":
    st.header("Clustering & Topic Modeling")
    try:
        with st.spinner("Fetching Clustering Data..."):
            res = requests.post(f"{API_BASE_URL}/cluster", json={"text": st.session_state["article_text"]})
            if res.status_code == 200:
                data = res.json()
                tab1, tab2 = st.tabs(["KMeans Clustering", "BERTopic Modeling"])
                
                with tab1:
                    st.metric("Predicted Cluster ID", data["kmeans"]["cluster_id"])
                    st.write("**Top Keywords:**", ", ".join(data["kmeans"]["labels"]))
                    
                    df_cl = pd.DataFrame([{
                        "Content": st.session_state["article_text"], 
                        "embedding": st.session_state["article_embedding"].tolist(), 
                        "cluster": data["kmeans"]["cluster_id"]
                    }])
                    kviz = KMeansVisualizer(df_cl)
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(kviz.plot_cluster_fit_gauge(), width='stretch')
                    with c2: st.plotly_chart(kviz.plot_cluster_keywords(), width='stretch')
                
                with tab2:
                    t_id = data["bertopic"]["topic_id"]
                    t_prob = data["bertopic"].get("topic_prob", [])
                    t_kw = data["bertopic"].get("topic_keywords", [])
                    
                    st.metric("Predicted BERTopic ID", t_id)
                    
                    c1, c2 = st.columns(2)
                    bviz = BERTopicVisualizer(
                        topic_id=t_id,
                        topic_prob=t_prob,
                        topic_keywords=t_kw
                    )
                    
                    with c1:
                        st.subheader("Topic Distribution")
                        dist_fig = bviz.plot_topic_distribution()
                        if dist_fig:
                            st.plotly_chart(dist_fig, width='stretch')
                        else:
                            st.info("No probability distribution available.")
                            
                    with c2: 
                        st.subheader("Topic Keywords")
                        kw_fig = bviz.plot_topic_keywords()
                        if kw_fig:
                            st.plotly_chart(kw_fig, width='stretch')
                        else:
                            st.info("No keywords found.")
            else:
                st.error(res.text)
    except Exception as e:
        st.error(f"Error: {e}")


elif page == "4. Named Entity Recognition":
    st.header("NER Analysis")
    try:
        with st.spinner("Extracting Entities..."):
            res = requests.post(f"{API_BASE_URL}/ner", json={"text": st.session_state["article_text"]})
            if res.status_code == 200:
                data = res.json()
                t1, t2, t3 = st.tabs(["spaCy", "GLiNER", "KeyBERT (Keywords)"])
                
                with t1:
                    st.components.v1.html(f"<div style='background-color: white; color: black; padding: 10px; border-radius: 5px;'>{data['spacy_html']}</div>", height=400, scrolling=True)
                    nviz = NERVisualizer(extraction_result=data["raw_ner_res"])
                    st.plotly_chart(nviz.plot_entity_distribution_spacy(), width='stretch')
                
                with t2:
                    st.components.v1.html(f"<div style='background-color: white; color: black; padding: 10px; border-radius: 5px;'>{data['gliner_html']}</div>", height=400, scrolling=True)
                    nviz2 = NERVisualizer(extraction_result=data["raw_ner_res"])
                    st.plotly_chart(nviz2.plot_entity_distribution_gliner(), width='stretch')
                
                with t3:
                    df_kw = pd.DataFrame(data["keywords"], columns=["Keyword", "Score"])
                    st.dataframe(df_kw.style.background_gradient(cmap="Blues", subset=['Score']))
                    nviz3 = NERVisualizer(extraction_result=data["raw_ner_res"])
                    st.plotly_chart(nviz3.plot_keywords(), width='stretch')
            else:
                st.error(res.text)
    except Exception as e:
        st.error(f"NER Error: {e}")


# --- Similar Articles (Global) ---
if page != "Home":
    st.divider()
    st.subheader("🔍 Similar Articles (via FAISS)")
    try:
        res = requests.post(f"{API_BASE_URL}/search", json={"text": st.session_state["article_text"]})
        if res.status_code == 200:
            data = res.json()
            st.write(f"*Routed via Topic #{data['topic']}*")
            for item in data["results"]:
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(f"**{item['Title']}**")
                        st.caption(f"Score: {item['Score']:.3f} | Cluster: {item['cluster']}")
                        st.write(item['Content'][:250] + "...")
                    with c2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Analyze", key=f"api_sim_{item['idx']}"):
                            with st.spinner("Processing suggested article..."):
                                try:
                                    response = requests.post(f"{API_BASE_URL}/process", json={"text": item["Content"]})
                                    if response.status_code == 200:
                                        data = response.json()
                                        st.session_state["article_text"] = item["Content"]
                                        st.session_state["word_count"] = len(item["Content"].split())
                                        feat_df = pd.DataFrame([data["features"]])
                                        feat_df['Content'] = item["Content"]
                                        st.session_state["features_df"] = feat_df
                                        st.session_state["article_embedding"] = np.array(data["embedding"])
                                        st.rerun()
                                    else:
                                        st.error(f"API Error: {response.text}")
                                except Exception as e:
                                    st.error(f"Processing Failed: {e}")
        else:
            st.info("Similar articles lookup failed or dataset not found on backend.")
    except Exception as e:
        st.error(f"Search API Error: {e}")

