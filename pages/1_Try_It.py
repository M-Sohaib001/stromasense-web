"""
Try It — the page that matters most. See ROADMAP.md Day 2.

Two curated modes (in-distribution / cross-domain), each running the real
pipeline live. Open upload is intentionally NOT the default interaction —
see the conversation this scaffold came out of for why: an unrestricted
upload box is the easiest way for this demo to fail unexplained in front of
judges. If you add it, keep it clearly gated (see the collapsed section at
the bottom) and never show a bare confidence number without the honesty
flag from src/utils.confidence_label.
"""

import json
import os

import streamlit as st
from PIL import Image

from src.components import disclaimer_banner, dataset_attribution_footer
from src.config import SAMPLE_IMAGES_DIR
from src.model_loader import load_all_models
from src.inference import run_pipeline
from src.utils import confidence_label

st.set_page_config(page_title="Try It — StromaSense", page_icon="🔬", layout="wide")
disclaimer_banner()
st.title("Try It")


@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_models():
    return load_all_models(device="cpu")


def load_manifest(subfolder: str) -> list:
    """
    manifest.json format (see assets/sample_images/README.md):
    [{"filename": "sample_01.png", "ground_truth": "Benign", "source": "BreaKHis"}, ...]
    """
    manifest_path = os.path.join(SAMPLE_IMAGES_DIR, subfolder, "manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        return json.load(f)


def render_result(image: Image.Image, result: dict, ground_truth: str = None):
    cols = st.columns(4)

    with cols[0]:
        st.image(image, caption="Input", use_container_width=True)

    if not result["roi_found"]:
        st.warning(result["warning"])
        st.caption(f"Ran in {result['inference_time_seconds']:.2f}s (CPU) before stopping at the ROI gate.")
        return

    with cols[1]:
        st.image(
            result["original_image_with_box"],
            caption=f"YOLO region (conf {result['roi_confidence']:.2f})",
            use_container_width=True,
        )
    with cols[2]:
        st.image(
            result["gradcam_overlay"],
            caption="Grad-CAM (ResNet50)",
            help="Highlights which pixels most influenced the ResNet50 branch's decision — brighter = more influence.",
            use_container_width=True,
        )
    with cols[3]:
        st.image(
            result["attention_overlay"],
            caption="Attention Rollout (ViT)",
            help="Shows which image regions the Vision Transformer's attention flowed to most strongly across all its layers.",
            use_container_width=True,
        )

    st.markdown(f"### Prediction: **{result['predicted_class']}**")
    st.markdown(
        f"Confidence: **{result['predicted_confidence']:.1%}** — "
        f"{confidence_label(result['predicted_confidence'], result['roi_found'])}"
    )
    if ground_truth:
        correct = ground_truth.lower() == result["predicted_class"].lower()
        st.markdown(f"Ground truth: **{ground_truth}** — {'✅ correct' if correct else '❌ incorrect'}")

    st.caption(f"⏱ Predicted in {result['inference_time_seconds']:.2f}s (CPU) — full pipeline, no GPU needed.")


tab_indist, tab_crossdomain = st.tabs(
    ["🟢 Sample Gallery (In-Distribution)", "🔶 Cross-Domain Challenge (BACH)"]
)

models = get_models()

with tab_indist:
    st.caption(
        "Held-out BreaKHis images — the distribution this model was trained and "
        "validated on. Expect strong, consistent results here."
    )
    samples = load_manifest("breakhis")
    if not samples:
        st.info(
            "No sample images yet. Add held-out BreaKHis images to "
            "assets/sample_images/breakhis/ and list them in manifest.json — "
            "see ROADMAP.md Day 2."
        )
    else:
        choice = st.selectbox(
            "Pick a sample:",
            options=[s["filename"] for s in samples],
            key="indist_choice",
        )
        sample = next(s for s in samples if s["filename"] == choice)
        img_path = os.path.join(SAMPLE_IMAGES_DIR, "breakhis", choice)
        image = Image.open(img_path)
        result = run_pipeline(image, models)
        render_result(image, result, ground_truth=sample.get("ground_truth"))

with tab_crossdomain:
    st.caption(
        "BACH images — a **different dataset, lab, and staining protocol** the "
        "model has never seen. Some of these are expected to fail or return "
        "'no ROI detected'. That's the point: this is our own measured "
        "generalization gap, shown rather than hidden. See Results & Limitations "
        "for the full breakdown."
    )
    samples = load_manifest("bach_challenge")
    if not samples:
        st.info(
            "No sample images yet. Add curated BACH images (include some 'normal' "
            "class ones — those are expected to fail, and that failure is the "
            "actual finding) to assets/sample_images/bach_challenge/ and list "
            "them in manifest.json."
        )
    else:
        choice = st.selectbox(
            "Pick a sample:",
            options=[s["filename"] for s in samples],
            key="crossdomain_choice",
        )
        sample = next(s for s in samples if s["filename"] == choice)
        img_path = os.path.join(SAMPLE_IMAGES_DIR, "bach_challenge", choice)
        image = Image.open(img_path)
        result = run_pipeline(image, models, apply_stain_norm=True)
        render_result(image, result, ground_truth=sample.get("ground_truth"))

st.markdown("---")
with st.expander("⚠️ Experimental: upload your own image (uncalibrated, use with caution)"):
    st.caption(
        "This model was validated only on BreaKHis and BACH. Any other image — "
        "different stain, different scanner, a phone photo, a non-breast-tissue "
        "image — is genuinely out of distribution in ways we haven't measured. "
        "Treat any result here as unverified."
    )
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded:
        image = Image.open(uploaded)
        result = run_pipeline(image, models)
        render_result(image, result)

dataset_attribution_footer()