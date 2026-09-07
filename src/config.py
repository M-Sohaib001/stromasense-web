"""
Central configuration. This is the first file to edit before anything else
in this repo will run.
"""

import os

# ---------------------------------------------------------------------------
# Hugging Face Hub model repo — where the actual weights live.
# Weights are NOT committed to this GitHub repo (see .gitignore) — they are
# downloaded at app startup from a HF Hub *model* repo you create separately.
# ---------------------------------------------------------------------------
HF_MODEL_REPO_ID = os.environ.get("STROMASENSE_HF_REPO", "Sohaib-001/stromasense-patient-split")

# Filenames inside that HF repo
WEIGHT_FILES = {
    "yolo": "best.pt",
    "resnet_classifier": "best_classifier.pt",
    "resnet_feature_extractor": "best_classifier.pt",
    "vit_feature_extractor": "vit_best.pt",
    "xgboost": "xgboost_fused.json",
    "xgboost_config": "xgb_config.json",
}

# ViT architecture — MUST match Part 2's training call exactly
# (google/vit-base-patch16-224, NOT the -in21k variant) or the state dict
# won't line up.
VIT_MODEL_NAME = "google/vit-base-patch16-224"
VIT_NUM_LABELS = 2

# Local cache dir for downloaded weights (created automatically at runtime;
# gitignored — never commit this directory).
MODEL_CACHE_DIR = os.environ.get("STROMASENSE_MODEL_DIR", "models")

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------
IMG_SIZE = 224                  # ResNet/ViT input size
YOLO_CONF_THRESHOLD = 0.25      # matches what you used during evaluation
CLASS_NAMES = ["Benign", "Malignant"]

# Normalization confirmed to match your Part 2/3/4 training code exactly
# (mean/std below are copied verbatim from your eval_transform).
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
# Confirmed correct against your training code — best_classifier.pt is a
# standard, unmodified torchvision resnet50, so "layer4" is right.
GRADCAM_TARGET_LAYER = "layer4"

# ---------------------------------------------------------------------------
# Contact — used on the Roadmap page's partnership CTA and in the Model Card.
# TODO: replace with a real, monitored address before you submit anything.
# ---------------------------------------------------------------------------
CONTACT_EMAIL = "muhammad.sohaib.2603@gmail.com"
TEAM_NAME = "StromaSense"

# ---------------------------------------------------------------------------
# Disclaimer — shown on every page. Do not remove or water this down.
# ---------------------------------------------------------------------------
DISCLAIMER_TEXT = (
    "**Research prototype — not a diagnostic device.** Built for an academic "
    "final-year project and hackathon submission. Not validated for "
    "real clinical use yet."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ASSETS_DIR = "assets"
SAMPLE_IMAGES_DIR = os.path.join(ASSETS_DIR, "sample_images")
METRICS_PATH = os.path.join("data", "metrics.json")
