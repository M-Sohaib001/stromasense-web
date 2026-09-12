"""
Full pipeline orchestration: image in -> (YOLO crop) -> fused features ->
XGBoost prediction -> Grad-CAM + attention rollout overlays.

This is the module Day 1 of ROADMAP.md asks you to test standalone, from a
plain script, before any Streamlit UI touches it.
"""

import time

import numpy as np
import torch
from PIL import Image

from src.config import YOLO_CONF_THRESHOLD, CLASS_NAMES, GRADCAM_TARGET_LAYER
from src.utils import pil_to_tensor, draw_bbox, overlay_heatmap
from src.explainability import GradCAM, attention_rollout


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize — used per-branch (ResNet, ViT) separately,
    matching your Part 4 eval code, never on the already-concatenated vector."""
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.clip(norm, 1e-12, None)


def _run_yolo(yolo_model, image: Image.Image):
    """
    Runs the ROI-gatekeeper YOLO model. Returns (crop, box, confidence) or
    (None, None, 0.0) if nothing was detected above threshold — this WILL
    happen often on out-of-distribution input, handle it, don't let it
    surface as a crash. See README "Known limitations".
    """
    results = yolo_model.predict(image, conf=YOLO_CONF_THRESHOLD, verbose=False)
    result = results[0]

    if len(result.boxes) == 0:
        return None, None, 0.0

    # Highest-confidence box, since this is meant to gatekeep the whole
    # image rather than find multiple lesions.
    best_idx = result.boxes.conf.argmax().item()
    box = result.boxes.xyxy[best_idx].tolist()
    conf = result.boxes.conf[best_idx].item()

    crop = image.crop(box)
    return crop, box, conf


def run_pipeline(image: Image.Image, models: dict, apply_stain_norm: bool = False) -> dict:
    """
    models: the dict returned by src.model_loader.load_all_models().

    apply_stain_norm: only pass True for a caller that has actually verified
    the effect on its own data (currently: the BACH cross-domain tab).
    Defaults to False since Macenko was never validated on BreaKHis.

    Returns a dict with everything a UI page needs to render one full
    result:
        roi_found (bool), roi_box, roi_confidence,
        predicted_class (str), predicted_confidence (float),
        original_image_with_box (PIL.Image or None),
        gradcam_overlay (PIL.Image), attention_overlay (PIL.Image),
        stain_normalized (bool or None), warning (str or None)
    """
    device = models["device"]
    image = image.convert("RGB")
    start_time = time.perf_counter()

    crop, box, roi_conf = _run_yolo(models["yolo"], image)

    if crop is None:
        return {
            "roi_found": False,
            "roi_box": None,
            "roi_confidence": 0.0,
            "predicted_class": None,
            "predicted_confidence": 0.0,
            "original_image_with_box": None,
            "gradcam_overlay": None,
            "attention_overlay": None,
            "stain_normalized": None,
            "inference_time_seconds": time.perf_counter() - start_time,
            "warning": (
                "No region of interest detected above the confidence "
                f"threshold ({YOLO_CONF_THRESHOLD}). This is expected and "
                "common on images outside the training distribution — see "
                "'Known limitations' on the Results page."
            ),
        }

    input_tensor, stain_normalized = pil_to_tensor(crop, apply_stain_norm=apply_stain_norm)
    input_tensor = input_tensor.to(device)

    # --- Fused features -> XGBoost prediction --------------------------------
    # resnet_features now returns (1, 2048, 1, 1) — flatten(1) instead of the
    # naive .squeeze() your eval script uses, so a batch size of 1 doesn't
    # also get squeezed away by accident.
    with torch.no_grad():
        resnet_feat = models["resnet_features"](input_tensor).flatten(1).cpu().numpy()
        # vit_features is the FULL ViTForImageClassification wrapper — call
        # its .vit(...) submodule, same as your Part 3/4 eval code, and use
        # last_hidden_state's CLS token, NOT pooler_output.
        vit_out = models["vit_features"].vit(input_tensor, output_attentions=True)
        vit_feat = vit_out.last_hidden_state[:, 0, :].cpu().numpy()

    if models["use_l2_normalization"]:
        resnet_feat = _l2_normalize(resnet_feat)
        vit_feat = _l2_normalize(vit_feat)

    fused = np.concatenate([resnet_feat, vit_feat], axis=1)  # (1, 2816)

    proba = models["xgboost"].predict_proba(fused)[0]
    predicted_idx = int(np.argmax(proba))
    predicted_class = CLASS_NAMES[predicted_idx]
    predicted_confidence = float(proba[predicted_idx])

    # --- Grad-CAM (ResNet classifier branch, separate forward pass) ----------
    # Needs grad, so re-run with grad enabled specifically for this branch —
    # everything above stays under no_grad for speed since XGBoost doesn't
    # need it. Context manager guarantees hooks are removed after this block,
    # even if something above throws — required so hooks don't accumulate on
    # the shared model across every prediction served by a long-running app.
    input_tensor_grad = input_tensor.clone().requires_grad_(True)
    with GradCAM(models["resnet_classifier"], target_layer_name=GRADCAM_TARGET_LAYER) as gradcam:
        cam_map = gradcam(input_tensor_grad, target_class=predicted_idx)
    gradcam_overlay = overlay_heatmap(crop, cam_map)

    # --- Attention rollout (ViT branch, already computed above) -------------
    rollout_map = attention_rollout(vit_out.attentions)
    attention_overlay = overlay_heatmap(crop, rollout_map)

    return {
        "roi_found": True,
        "roi_box": box,
        "roi_confidence": roi_conf,
        "predicted_class": predicted_class,
        "predicted_confidence": predicted_confidence,
        "original_image_with_box": draw_bbox(image, box),
        "gradcam_overlay": gradcam_overlay,
        "attention_overlay": attention_overlay,
        "inference_time_seconds": time.perf_counter() - start_time,
        "warning": None,
        "stain_normalized": stain_normalized,
    }