"""
StromaSense — Home page.
Run with: streamlit run app.py
"""

import json
import streamlit as st

from src.components import disclaimer_banner
from src.config import METRICS_PATH

st.set_page_config(
    page_title="StromaSense",
    page_icon="🔬",
    layout="wide",
)

disclaimer_banner()

st.title("🔬 StromaSense")
st.subheader("Breast cancer histopathology classification — a research prototype")

st.markdown(
    """
StromaSense classifies breast tissue histopathology images as **benign** or
**malignant** using a cascaded YOLO → ResNet50+ViT (fused) → XGBoost
pipeline, with Grad-CAM and attention-rollout explainability on every
prediction.

This page gives the headline numbers. **Read the "Results & Limitations"
page before drawing conclusions from them** — the short version is that this
model is strong on the data it was trained on and does not yet generalize to
data from other labs/scanners, and we think that gap is the most important
finding in this whole project, not something to gloss over.
"""
)

# --- Architecture diagram -----------------------------------------------
st.markdown("### Pipeline")
st.markdown(
    "`Image → YOLO (ROI gate) → ResNet50 + ViT-Base (fused features) → "
    "XGBoost → prediction + Grad-CAM + attention rollout`"
)
# TODO: replace with st.image("assets/diagrams/architecture.png") once you've
# dropped your actual diagram in place.

# --- Headline metrics -----------------------------------------------------
st.markdown("### Headline results")

try:
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "BreaKHis accuracy (patient-split, in-distribution)",
            f"{metrics['headline']['breakhis_patient_split']['accuracy']:.1%}",
        )
    with col2:
        st.metric(
            "BACH accuracy (zero-shot, external)",
            f"{metrics['headline']['bach_zero_shot']['accuracy']:.1%}",
            help="Measured on a dataset from a different lab/scanner, never seen during training.",
        )

    st.caption(
        "The gap between these two numbers is expected and diagnosed, not a bug — "
        "see Results & Limitations for why, and How It Works for the patient-split "
        "vs. image-split leakage story."
    )
except FileNotFoundError:
    st.info("data/metrics.json not found yet — fill it in per ROADMAP.md Day 3.")

st.markdown("---")
st.markdown(
    "📄 Paper: _add your IEEE Access link_ &nbsp;·&nbsp; "
    "💻 Code: _add your GitHub link_"
)
