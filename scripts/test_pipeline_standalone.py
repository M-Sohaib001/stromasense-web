"""
Day 1 sanity check (see ROADMAP.md). Run this BEFORE touching any Streamlit
code:

    python scripts/test_pipeline_standalone.py path/to/a/known_benign_image.png

It prints the prediction and saves gradcam.png / attention_rollout.png next
to wherever you run it from. If this doesn't work, the UI won't either —
fix it here first, where a traceback is easy to read.
"""

import sys
from PIL import Image

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_loader import load_all_models
from src.inference import run_pipeline


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Loading models (this pulls weights from HF Hub on first run)...")
    models = load_all_models(device="cpu")

    print(f"Running pipeline on {image_path} ...")
    image = Image.open(image_path)
    result = run_pipeline(image, models)

    if not result["roi_found"]:
        print(f"NO ROI DETECTED: {result['warning']}")
        return

    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence:      {result['predicted_confidence']:.4f}")
    print(f"ROI confidence:  {result['roi_confidence']:.4f}")

    result["gradcam_overlay"].save("gradcam_output.png")
    result["attention_overlay"].save("attention_rollout_output.png")
    print("Saved gradcam_output.png and attention_rollout_output.png")


if __name__ == "__main__":
    main()
