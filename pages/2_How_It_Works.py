import streamlit as st
from src.components import disclaimer_banner

st.set_page_config(page_title="How It Works — StromaSense", page_icon="⚙️", layout="wide")
disclaimer_banner()
st.title("How It Works")

st.markdown(
    """
### The pipeline

1. **YOLO (ROI gate).** Trained with whole-image bounding boxes, so its
   actual role is closer to "does this look like a valid input image" than
   true lesion localization. This matters: it's also the stage that fails
   most often on out-of-distribution images (see Results & Limitations).

2. **ResNet50 + ViT-Base (parallel feature extractors).** Each produces a
   feature vector from the YOLO crop — 2048-D from ResNet50, 768-D from
   ViT-Base — concatenated into a single 2816-D representation.

3. **XGBoost.** Takes the fused 2816-D vector and outputs Benign/Malignant
   with a confidence score.

4. **Explainability, computed separately:**
   - **Grad-CAM**, run against a *separately trained* ResNet50 classifier
     head (not the feature-extractor variant above — Grad-CAM needs a
     differentiable class score to backprop through, which the fusion
     branch doesn't have on its own since XGBoost isn't part of that
     computational graph).
   - **Attention Rollout**, computed directly from the ViT's own attention
     weights during the same forward pass used for feature extraction.

### Why we deploy the patient-split model, not the higher-scoring one

An earlier version of this pipeline split BreaKHis into train/test **by
image**, not by patient. Because BreaKHis contains multiple crops per
patient, that split let visually correlated crops from the same patient
appear in both train and test — inflating the apparent score. We measured
the effect directly: Matthews Correlation Coefficient dropped from **0.96 to
0.88** once the split was redone by patient ID. The number this app reports
is the patient-split one, because it's the one we trust to mean what it
looks like it means.

### What this pipeline is not

It is not a tumor segmentation system — the YOLO stage does not localize
lesions within tissue, it gates whether the input looks like a valid image
at all. It is not clinically validated. See Results & Limitations for the
external validation findings that define the actual current scope of this
tool.
"""
)
