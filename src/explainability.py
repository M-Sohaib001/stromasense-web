"""
Grad-CAM (ResNet50 branch) and Attention Rollout (ViT branch).

These are two different explainability techniques on two different
architectures, which is why the pipeline produces two separate heatmaps
rather than one combined view — the ResNet50 map shows convolutional
activation strength, the ViT map shows attention flow. Show both,
don't average them into one number; they're not measuring the same thing.
"""

import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """
    Standard Grad-CAM (Selvaraju et al., 2017) against the ResNet50
    *classifier* variant (the one with its own softmax head — see
    model_loader.load_resnet_classifier). Runs on the whole crop, hooked at
    the target conv layer (default: layer4, the last conv block).

    FIX: hooks are registered in __init__ but must be removed after use, or
    they accumulate on the shared model across every prediction served by
    the app (each GradCAM() call would otherwise add 2 more hooks forever).
    Use as a context manager so cleanup is guaranteed even on exception:

        with GradCAM(model, "layer4") as cam:
            heatmap = cam(input_tensor, target_class=predicted_idx)
        # hooks removed automatically here

    If you must keep the object alive longer, call .remove_hooks() yourself.
    """

    def __init__(self, model, target_layer_name: str = "layer4"):
        self.model = model
        self.gradients = None
        self.activations = None
        self._handles = []

        target_layer = dict(model.named_modules())[target_layer_name]
        self._handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Remove all registered hooks. Called automatically on context exit."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def __call__(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        input_tensor: (1, 3, H, W), already normalized.
        target_class: index into the classifier's output logits. If None,
        uses the model's own predicted class (argmax).
        Returns a (H, W) heatmap normalized to [0, 1], resized to input size.
        """
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[:, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def attention_rollout(
    attentions: list,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> np.ndarray:
    """
    Attention Rollout (Abnar & Zuidema, 2020) for a ViT.

    attentions: list of per-layer attention tensors, each
        (batch, num_heads, seq_len, seq_len), as returned by
        ViTModel(..., output_attentions=True).attentions — this is why
        model_loader loads ViT with attn_implementation='eager', the
        fused/SDPA attention path does not return these.
    Returns a (grid_size, grid_size) heatmap (patch-level, excluding the
    CLS token), normalized to [0, 1].
    """
    result = torch.eye(attentions[0].size(-1))

    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                fused = attention.mean(dim=1)
            elif head_fusion == "max":
                fused = attention.max(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion: {head_fusion}")

            fused = fused[0]  # drop batch dim -> (seq_len, seq_len)

            if discard_ratio > 0:
                flat = fused.view(-1)
                num_discard = int(flat.size(0) * discard_ratio)
                _, idx = flat.topk(num_discard, largest=False)
                flat[idx] = 0
                fused = flat.view(fused.shape)

            # Add identity for the residual connection, then re-normalize
            # rows to sum to 1 — this is the step that makes "rollout"
            # different from just multiplying raw attention matrices.
            identity = torch.eye(fused.size(-1))
            fused = (fused + identity) / 2
            fused = fused / fused.sum(dim=-1, keepdim=True)

            result = torch.matmul(fused, result)

    # Row 0 = CLS token's rolled-up attention to every patch token
    cls_attention = result[0, 1:]
    grid_size = int(cls_attention.size(0) ** 0.5)
    heatmap = cls_attention.reshape(grid_size, grid_size).numpy()

    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    return heatmap
