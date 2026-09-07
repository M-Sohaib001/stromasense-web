"""Image preprocessing and visualization helpers shared across pages."""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from src.config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def get_inference_transform() -> transforms.Compose:
    """
    The exact preprocessing every crop goes through before hitting
    ResNet50/ViT. TODO: this MUST match your training-time transform
    exactly (same resize, same normalization) — a silent mismatch here
    won't error, it'll just quietly degrade every prediction.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL image -> normalized (1, 3, IMG_SIZE, IMG_SIZE) tensor."""
    transform = get_inference_transform()
    tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0)


def draw_bbox(image: Image.Image, box: tuple, color=(255, 0, 0), width: int = 4) -> Image.Image:
    """Draws a single YOLO box (x1, y1, x2, y2) on a copy of the image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(box, outline=color, width=width)
    return img_copy


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlays a normalized [0, 1] heatmap (any spatial size — will be
    resized) onto a PIL image using a standard jet colormap. Used for both
    Grad-CAM and (upsampled) attention rollout maps so the two look visually
    consistent side by side, even though they come from different methods.
    """
    img_np = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = (1 - alpha) * img_np + alpha * heatmap_color
    overlaid = np.uint8(np.clip(overlaid, 0, 255))
    return Image.fromarray(overlaid)


def confidence_label(confidence: float, roi_found: bool) -> str:
    """
    Turns a bare confidence number into an honest qualitative flag, per the
    "don't show a bare percentage" rule — see ROADMAP.md Day 5.
    """
    if not roi_found:
        return "⚠️ No region of interest detected — result not shown"
    if confidence >= 0.80:
        return "High confidence"
    if confidence >= 0.60:
        return "Moderate confidence"
    return "Low confidence — treat this prediction with real skepticism"
