import random
from pathlib import Path

import streamlit as st

from model import detokenize, generate, load_json

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "catbot_model.json"
LABEL = "AI-generated text — this is not a real quote from the speaker."

st.set_page_config(page_title="CATbot", page_icon="🐱", layout="centered")
st.title("🐱 CATbot — Copy-cat Text Generator")
st.caption("A Markov-chain language model that learns word-transition patterns from your transcript dataset.")

@st.cache_resource
def load_model(path, mtime):
    return load_json(path)

if not MODEL_PATH.exists():
    st.warning("No trained model found yet. Train in Google Colab, download catbot_model.json, and place it in the project's model folder.")
    st.code(r"CATbot_Complete\model\catbot_model.json")
    st.stop()

data, transitions, starts = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
with st.sidebar:
    st.header("Model information")
    st.write(f"**Type:** {data.get('model_type', 'Markov chain')}")
    st.write(f"**Training words:** {data.get('word_count', 0):,}")
    st.write(f"**States:** {len(transitions):,}")
    st.write(f"**Memory order:** {data.get('order', 2)}")
    st.write("**Sources:**")
    for name in data.get("source_files", []): st.caption(name)
    st.divider()
    st.caption("Generated text is synthetic and may be inaccurate or nonsensical.")

st.subheader("Generate text")
with st.form("generate_form"):
    start = st.text_input("Starting phrase (optional)", placeholder="e.g. my dear")
    col1, col2 = st.columns(2)
    with col1: length = st.slider("Output length (tokens)", 30, 300, 100, 10)
    with col2: seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
    submitted = st.form_submit_button("Generate CATbot text", type="primary", use_container_width=True)
if submitted:
    try:
        tokens = generate(transitions, starts, data.get("order", 2), start, length, random.Random(int(seed)))
        st.markdown("### CATbot response")
        st.write(detokenize(tokens))
        st.info(LABEL)
        st.download_button("Download generated text", detokenize(tokens) + "\n\n" + LABEL, file_name="catbot_output.txt", mime="text/plain")
    except Exception as e:
        st.error(str(e))
