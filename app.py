"""
FootballSearch — Streamlit demo
Run locally with:  streamlit run app.py
"""

import sys
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Path Resolution & Imports
# ---------------------------------------------------------------------------
# Dynamically resolve the absolute path to the directory containing this script.
BASE_DIR = Path(__file__).resolve().parent

# Add the base directory to sys.path so 'src' can be imported regardless of execution directory.
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Fallback block in case 'src' is missing during testing
try:
    from src.data_utils import combine_title_body  
except ImportError:
    st.warning("Could not import `src.data_utils`. Using a basic fallback for `combine_title_body`.")
    def combine_title_body(title, body):
        return f"{title} {body}" if pd.notna(body) else str(title)

# ---------------------------------------------------------------------------
# Config — Resolved dynamically against BASE_DIR
# ---------------------------------------------------------------------------
QUESTIONS_CSV = BASE_DIR / "data" / "processed" / "questions.csv"
METRICS_JSON = BASE_DIR / "results" / "metrics.json"

VARIANTS = {
    "Baseline (all-mpnet-base-v2)": {
        "model_path": "all-mpnet-base-v2", # HuggingFace hub model (leave as string)
        "metrics_key": "baseline",
    },
    "Fine-tuned (all pairs)": {
        "model_path": str(BASE_DIR / "models" / "ft-all-pairs"), # Local model (absolute path)
        "metrics_key": "ft-all-pairs",
    },
    # Add more variants here as you finalize the hard-negatives run, e.g.:
    # "Fine-tuned (hard negatives)": {
    #     "model_path": str(BASE_DIR / "models" / "ft-hard-negatives"),
    #     "metrics_key": "ft-hard-negatives",
    # },
}

TOP_K = 5

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_questions():
    if not QUESTIONS_CSV.exists():
        st.error(f"**Data Error:** Could not find {QUESTIONS_CSV}. Please ensure your data is generated.")
        st.stop()
        
    df = pd.read_csv(QUESTIONS_CSV)
    df["combined"] = df.apply(
        lambda r: combine_title_body(r["Title"], r.get("Body", "")), axis=1
    )
    return df


@st.cache_resource
def load_metrics():
    if not METRICS_JSON.exists():
        return {}
    return json.loads(METRICS_JSON.read_text())


@st.cache_resource
def load_model_and_index(variant_label: str, _questions_df: pd.DataFrame):
    """Loads model, encodes corpus (or loads cached .npy), builds a FAISS IndexFlatIP."""
    variant = VARIANTS[variant_label]
    
    # Handle local vs huggingface paths safely
    model_path = variant["model_path"]
    if "models/" in model_path and not Path(model_path).exists():
        st.error(f"**Model Error:** Could not find model at {model_path}. Did you run the fine-tuning script?")
        st.stop()
        
    model = SentenceTransformer(model_path)

    # Use BASE_DIR for caching embeddings
    cache_path = BASE_DIR / "data" / "processed" / f"embeddings_{variant['metrics_key']}.npy"
    
    if cache_path.exists():
        embeddings = np.load(cache_path)
    else:
        with st.spinner(f"Generating embeddings for {variant_label}... (This might take a minute)"):
            embeddings = model.encode(
                _questions_df["combined"].tolist(),
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return model, index


def search(query: str, model, index, questions_df: pd.DataFrame, top_k: int = TOP_K):
    query_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(query_emb.astype("float32"), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = questions_df.iloc[idx]
        results.append({"title": row["Title"], "score": float(score), "id": row.get("Id", idx)})
    return results


# ---------------------------------------------------------------------------
# UI - Aesthetic Upgrade
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FootballSearch", page_icon="⚽", layout="wide")

# Inject Custom CSS for a polished, modern dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 10px;
        border: 2px solid #1DB954; /* Pitch Green */
        padding: 10px;
        font-size: 16px;
    }
    .result-card {
        background-color: #1A1C23;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #1DB954;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .score-badge {
        background-color: #2D333B;
        color: #1DB954;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Layout: Header
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/5328/5328087.png", width=70) # Generic football icon
with col2:
    st.title("FootballSearch")
    st.markdown("*AI-powered semantic search over Sports Stack Exchange.*")

questions_df = load_questions()
metrics = load_metrics()

# Layout: Sidebar for controls
with st.sidebar:
    st.header("⚙️ Settings")
    compare_mode = st.toggle("Compare baseline vs fine-tuned", value=True)
    if not compare_mode:
        selected_model = st.selectbox("Select Model", list(VARIANTS.keys()))
    
    st.divider()
    with st.expander("📊 Evaluation Metrics", expanded=False):
        if metrics:
            for label, variant in VARIANTS.items():
                m = metrics.get(variant["metrics_key"], {})
                if m:
                    st.markdown(f"**{label}**")
                    st.caption(f"Recall@5: {m.get('recall@5', 0):.3f} | MRR: {m.get('mrr', 0):.3f}")
        else:
            st.caption("Metrics not found.")

# Layout: Main Search Area
st.markdown("<br>", unsafe_allow_html=True)
query = st.text_input("Ask a football question:", placeholder="e.g., What is a false 9 tactic?")

if query:
    st.markdown("### Results")
    if compare_mode:
        cols = st.columns(len(VARIANTS))
        for col, label in zip(cols, VARIANTS.keys()):
            with col:
                st.markdown(f"**{label}**")
                model, index = load_model_and_index(label, questions_df)
                results = search(query, model, index, questions_df)
                for r in results:
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{r['title']}</h4>
                        <span class="score-badge">Score: {r['score']:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        model, index = load_model_and_index(selected_model, questions_df)
        results = search(query, model, index, questions_df)
        
        # Trigger a fun effect if we find a very high-confidence match!
        if results and results[0]['score'] > 0.85:
            st.balloons() 

        for r in results:
            st.markdown(f"""
            <div class="result-card">
                <h4>{r['title']}</h4>
                <span class="score-badge">Score: {r['score']:.3f}</span>
            </div>
            """, unsafe_allow_html=True)