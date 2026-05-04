"""
Streamlit Demo - Multilingual NMT with mBART-50
"""

import streamlit as st
import torch
import json
import pandas as pd
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.inference.translator import MultilingualTranslator

st.set_page_config(
    page_title="mBART-50 NMT Demo",
    layout="wide"
)

# Supported language pairs from training
SUPPORTED_PAIRS = {
    "en-vi": {"src": "en", "tgt": "vi", "src_name": "English", "tgt_name": "Vietnamese"},
    "en-fr": {"src": "en", "tgt": "fr", "src_name": "English", "tgt_name": "French"},
    "de-en": {"src": "de", "tgt": "en", "src_name": "German", "tgt_name": "English"},
}


@st.cache_resource
def load_translator(model_path: str):
    return MultilingualTranslator(model_path)


def load_eval_metrics():
    eval_dir = Path("outputs/eval")
    if eval_dir.exists():
        metric_files = list(eval_dir.glob("*_metrics_*.json"))
        if metric_files:
            latest = max(metric_files, key=lambda p: p.stat().st_mtime)
            with open(latest, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_sample_predictions():
    eval_dir = Path("outputs/eval")
    if eval_dir.exists():
        pred_files = list(eval_dir.glob("*_preds_*.csv"))
        if pred_files:
            latest = max(pred_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_csv(latest)
            return df.head(20)
    return None


def run_strategy_comparison(translator, text: str, src_lang: str, tgt_lang: str, max_length: int = 128):
    strategies = {
        "Greedy": {"num_beams": 1, "do_sample": False},
        "Beam Search (k=4)": {"num_beams": 4, "do_sample": False},
        "Top-k (k=50)": {"num_beams": 1, "do_sample": True, "top_k": 50, "temperature": 0.7},
        "Top-p (p=0.9)": {"num_beams": 1, "do_sample": True, "top_p": 0.9, "temperature": 0.7},
        "Top-k+p": {"num_beams": 1, "do_sample": True, "top_k": 50, "top_p": 0.9, "temperature": 0.7},
    }

    results = []
    for name, params in strategies.items():
        start_time = time.time()
        try:
            translation = translator.translate(text, src_lang, tgt_lang, max_length=max_length, **params)
            elapsed = time.time() - start_time
            results.append({"Strategy": name, "Translation": translation, "Time (s)": f"{elapsed:.2f}"})
        except Exception as e:
            results.append({"Strategy": name, "Translation": f"Error: {str(e)}", "Time (s)": "-"})
    return results


# ============================================================
# Main
# ============================================================
st.title("mBART-50 Multilingual NMT Demo")

# Load model
model_options = {
    "Fine-tuned": "outputs/best_model",
    "Original mBART-50": "facebook/mbart-large-50-many-to-many-mmt"
}
model_choice = st.selectbox("Model", list(model_options.keys()))
model_path = model_options[model_choice]

try:
    translator = load_translator(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Translation", "BLEU / chrF++", "Strategy Comparison", "Sample Predictions"])

# ------------------------------------------------------------
# Tab 1: Translation
# ------------------------------------------------------------
with tab1:
    st.header("Translate")

    pair_options = list(SUPPORTED_PAIRS.keys())
    selected_pair = st.selectbox("Language Pair", pair_options)

    pair_info = SUPPORTED_PAIRS[selected_pair]
    src_lang = pair_info["src"]
    tgt_lang = pair_info["tgt"]

    col1, col2 = st.columns(2)
    with col1:
        st.text(f"{pair_info['src_name']} ({src_lang})")
    with col2:
        st.text(f"{pair_info['tgt_name']} ({tgt_lang})")

    input_text = st.text_area("Input", height=100)

    strategy = st.selectbox(
        "Strategy",
        ["greedy", "beam", "top_k", "top_p"],
        index=1
    )

    if st.button("Translate", type="primary", use_container_width=True):
        if input_text.strip():
            with st.spinner("Translating..."):
                try:
                    translation = translator.translate_with_strategy(
                        input_text, src_lang, tgt_lang,
                        strategy=strategy, max_length=128
                    )
                    st.text_area("Output", value=translation, height=100)
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("Enter text to translate.")

# ------------------------------------------------------------
# Tab 2: BLEU / chrF++
# ------------------------------------------------------------
with tab2:
    st.header("BLEU / chrF++ Comparison")

    metrics = load_eval_metrics()

    if metrics:
        pairs = metrics.get("metrics_by_pair", {})
        overall = metrics.get("overall_metrics", {})

        st.subheader("Overall")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BLEU", f"{overall.get('bleu', 0):.2f}")
        with col2:
            st.metric("chrF++", f"{overall.get('chrf', 0):.2f}")

        st.subheader("By Language Pair")
        if pairs:
            data = []
            for pair, scores in pairs.items():
                data.append({
                    "Pair": pair.upper(),
                    "BLEU": round(scores.get("bleu", 0), 2),
                    "chrF++": round(scores.get("chrf", 0), 2)
                })
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.bar_chart(df.set_index("Pair")["BLEU"])
    else:
        st.info("No evaluation metrics found.")

# ------------------------------------------------------------
# Tab 3: Strategy Comparison
# ------------------------------------------------------------
with tab3:
    st.header("Strategy Comparison")

    pair_options = list(SUPPORTED_PAIRS.keys())
    selected_pair = st.selectbox("Language Pair", pair_options, key="comp_pair")

    pair_info = SUPPORTED_PAIRS[selected_pair]
    src_lang = pair_info["src"]
    tgt_lang = pair_info["tgt"]

    input_text = st.text_area("Input text", value="I am very happy to see you today!", height=80, key="comp_text")

    if st.button("Compare All Strategies", type="primary", use_container_width=True):
        if input_text.strip():
            with st.spinner("Running..."):
                results = run_strategy_comparison(translator, input_text, src_lang, tgt_lang)
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("Enter text.")

# ------------------------------------------------------------
# Tab 4: Sample Predictions
# ------------------------------------------------------------
with tab4:
    st.header("Sample Predictions")

    samples = load_sample_predictions()

    if samples is not None:
        st.text(f"Showing {len(samples)} samples")
        st.dataframe(samples, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions found.")
