"""
Loads all four pipeline components (YOLO, ResNet50 x2 variants, ViT,
XGBoost), downloading weights from a Hugging Face Hub model repo on first
run and caching them locally after that.

Streamlit note: every function that returns a loaded model is wrapped with
st.cache_resource by the caller in app.py / pages/*.py — NOT here, so this
module stays importable and testable outside Streamlit (see ROADMAP.md Day 1:
run this from a bare script before wiring up any UI).
"""

import os
import json
import torch
import torch.nn as nn
import xgboost as xgb
from torchvision import models as tv_models
from transformers import ViTForImageClassification
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from src.config import (
    HF_MODEL_REPO_ID,
    WEIGHT_FILES,
    MODEL_CACHE_DIR,
    VIT_MODEL_NAME,
    VIT_NUM_LABELS,
)

if int(transformers.__version__.split(".")[0]) >= 5:
    raise RuntimeError(
        f"transformers=={transformers.__version__} is installed, but this "
        f"project's ViT checkpoint requires transformers<5 (v5 renamed "
        f"ViT's internal module names, breaking old checkpoints). Run: "
        f'pip install "transformers<5"'
    )


def _get_weight_path(key: str) -> str:
    """
    Returns a local path to the requested weight file, downloading it from
    the HF Hub model repo if it isn't already cached locally.
    """
    filename = WEIGHT_FILES[key]
    local_path = os.path.join(MODEL_CACHE_DIR, filename)
    if os.path.exists(local_path):
        return local_path

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=filename,
            local_dir=MODEL_CACHE_DIR,
        )
        return downloaded_path
    except Exception as e:
        # Surface a clear message instead of a raw traceback — this is the
        # failure mode most likely to happen live in front of judges if HF
        # Hub is slow/unreachable, so don't let it crash silently.
        raise RuntimeError(
            f"Could not load model weight '{filename}' from HF repo "
            f"'{HF_MODEL_REPO_ID}'. Check src/config.py HF_MODEL_REPO_ID "
            f"and WEIGHT_FILES, and that the repo/file are public or you're "
            f"authenticated. Original error: {e}"
        )


def load_yolo() -> YOLO:
    """Loads the patient-split YOLO detector (whole-image ROI gatekeeper)."""
    weight_path = _get_weight_path("yolo")
    return YOLO(weight_path)


def load_resnet_classifier(num_classes: int = 2, device: str = "cpu") -> nn.Module:
    """
    Full ResNet50 classifier (with its own softmax head) — used ONLY for
    Grad-CAM, since Grad-CAM needs a differentiable class score to backprop
    against. Confirmed against your Part 2 training code: plain
    torchvision resnet50, fc replaced with Linear(in_features, 2).
    """
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    weight_path = _get_weight_path("resnet_classifier")
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_resnet_feature_extractor(device: str = "cpu") -> nn.Module:
    """
    Headless ResNet50 feature extractor. Rebuilt the way your own Part 3/4
    code does it: load the full 2-class model strictly (so any mismatch
    errors loudly instead of hiding), then slice off the fc layer with
    nn.Sequential — NOT swapped for nn.Identity with strict=False, which
    would mask a real architecture mismatch instead of catching it.

    Output shape is (batch, 2048, 1, 1) since this skips the explicit
    flatten() that resnet50's own .forward() normally does — callers must
    flatten(1) before using it (see src/inference.py).
    """
    base = tv_models.resnet50(weights=None)
    base.fc = nn.Linear(base.fc.in_features, 2)
    weight_path = _get_weight_path("resnet_feature_extractor")
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    base.load_state_dict(state_dict)  # strict — errors loudly on any mismatch
    extractor = nn.Sequential(*list(base.children())[:-1]).to(device).eval()
    return extractor


def load_vit_feature_extractor(device: str = "cpu") -> ViTForImageClassification:
    """
    Loads the FULL ViTForImageClassification wrapper — matching your Part 2
    training call exactly (google/vit-base-patch16-224, num_labels=2,
    ignore_mismatched_sizes=True) — NOT a bare ViTModel.

    This matters: vit_best.pt's state dict keys are prefixed 'vit.*' and
    'classifier.*' because it's a ViTForImageClassification checkpoint. A
    bare ViTModel's keys have no such prefix, so loading into one with
    strict=False silently matches nothing and leaves the model at its
    random/pretrained init with no error raised — the bug this replaces.

    Callers use the returned model's .vit(...) submodule for the actual
    forward pass (see src/inference.py), same as your Part 3/4 eval code.
    attn_implementation='eager' is required for attention rollout to see
    real per-layer attention weights.
    """
    model = ViTForImageClassification.from_pretrained(
        VIT_MODEL_NAME,
        num_labels=VIT_NUM_LABELS,
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )
    weight_path = _get_weight_path("vit_feature_extractor")
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)  # strict — architecture matches exactly, should never mismatch
    model.to(device).eval()
    return model


def load_xgboost() -> xgb.XGBClassifier:
    """
    Loads the fused-feature XGBoost classifier (2816-D input: 2048 + 768)
    from its native XGBoost JSON format (save_model() output).
    """
    weight_path = _get_weight_path("xgboost")
    model = xgb.XGBClassifier()
    model.load_model(weight_path)
    return model


def load_xgb_config() -> dict:
    """
    Loads the CUSTOM config JSON your Part 3 script writes alongside the
    XGBoost model — NOT xgboost's own save_config(). Contains
    'use_l2_normalization': whether ResNet and ViT feature blocks must be
    L2-normalized SEPARATELY before concatenation (see Part 4's
    extract_fused_features). Getting this flag wrong doesn't error — it
    silently feeds XGBoost the wrong-shaped-distribution features.
    """
    config_path = _get_weight_path("xgboost_config")
    with open(config_path) as f:
        cfg = json.load(f)
    if "use_l2_normalization" not in cfg:
        print(
            "WARNING: 'use_l2_normalization' not found in xgboost_config.json "
            "— defaulting to False. Verify this against your actual training "
            "run before trusting predictions."
        )
    return cfg


def load_all_models(device: str = "cpu") -> dict:
    """Convenience loader — returns a dict of every component, ready for
    src.inference.run_pipeline(). Wrap the *caller* of this in
    st.cache_resource so it only runs once per Space session."""
    xgb_cfg = load_xgb_config()
    return {
        "yolo": load_yolo(),
        "resnet_classifier": load_resnet_classifier(device=device),
        "resnet_features": load_resnet_feature_extractor(device=device),
        "vit_features": load_vit_feature_extractor(device=device),
        "xgboost": load_xgboost(),
        "use_l2_normalization": bool(xgb_cfg.get("use_l2_normalization", False)),
        "device": device,
    }
