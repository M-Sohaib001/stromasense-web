# StromaSense — Breast Cancer Histopathology Classification (Research Prototype)

> **Not a diagnostic device. Not for clinical use.** This is an academic /
> hackathon research prototype demonstrating a multi-model histopathology
> classification pipeline, including an honest, quantified account of where
> it does and does not generalize.

**Live demo:** _add your Hugging Face Space URL here once deployed_
**Paper:** _add your IEEE Access link here_
**Team:** _add names / contact here_

---

## Table of contents

- [What this is](#what-this-is)
- [What it hopes to solve](#what-it-hopes-to-solve)
- [Architecture](#architecture)
- [Results](#results)
- [Known limitations](#known-limitations)
- [Repo structure](#repo-structure)
- [Getting started](#getting-started)
- [Roadmap](#roadmap)
- [Datasets & citations](#datasets--citations)
- [License](#license)

---

## What this is

StromaSense is a cascaded pipeline for classifying breast tissue histopathology
images as **benign** or **malignant**:

```
Input image
    │
    ▼
 YOLO  ──────────────►  region-of-interest crop (or "no detection")
    │
    ▼
 ResNet50  +  ViT-Base  ──────►  2048-D + 768-D features, concatenated
    │
    ▼
 XGBoost  ──────────────►  Benign / Malignant + confidence
    │
    ▼
 Grad-CAM (ResNet branch) + Attention Rollout (ViT branch)
    ──────────────►  visual explanation of the prediction
```

Trained and validated on **BreaKHis** (7,909 microscopy images, 82 patients,
40×-400× magnification). Externally validated on **BACH** (ICIAR 2018
Challenge, 400 images, 4 classes) to test generalization to a dataset from a
different lab, scanner, and staining protocol.

## What it hopes to solve

Histopathology review is slow, expertise-bottlenecked, and subject to
inter-pathologist disagreement. StromaSense explores whether a fused
CNN+Transformer feature representation, paired with tree-based classification
and multi-layer explainability (so a prediction never arrives without a
visual "why"), can serve as a decision-support aid — while being explicit
about the point at which today's version stops being trustworthy.

## Architecture

See [`pages/2_How_It_Works.py`](pages/2_How_It_Works.py) for the full
walkthrough, or the diagram at `assets/diagrams/`.

Two things worth knowing up front, because they explain the results below:

- **YOLO here is a whole-image gatekeeper, not a lesion localizer.** Its
  training labels cover the entire image (`0 0.5 0.5 1.0 1.0`), so its actual
  job is "does this look like a valid input," not "where is the tumor."
- **We deploy the patient-level split model, not the higher-scoring
  image-level split model.** An image-level split lets crops from the same
  patient appear in both train and test, which inflates accuracy (we
  measured this: MCC dropped from 0.96 to 0.88 once patient IDs were split
  properly). The number below is the one we trust.

## Results

| Metric | BreaKHis (patient-split, in-distribution) | BACH (zero-shot, external) |
|---|---|---|
| Accuracy | 94.71% | ~54.75% |
| MCC | 0.88 | 0.12 |
| ROC-AUC | 0.974 | 0.559 |

**Per-class accuracy on BACH (patient-split model, classifier only):**

| Class | Accuracy | Note |
|---|---|---|
| In-situ carcinoma | 88% | Maps to "malignant" — matches training concept |
| Invasive carcinoma | 86% | Maps to "malignant" — matches training concept |
| Benign | 23% | Trained class, but visually distinct subtypes from BreaKHis |
| Normal | 22% | **Never present in training data at all** |

The pattern: **sensitivity generalizes** (the model reliably catches
malignant-type tissue across both datasets), **specificity does not** — the
model over-predicts "malignant" on tissue it wasn't trained to positively
recognize, whether that's a genuinely unseen concept (normal) or a seen
concept with different visual characteristics (benign, different subtypes
across the two datasets/populations).

Full breakdown, including the effect of stain normalization and a
multi-reference stability sweep, is on the Results & Limitations page of the
live app.

## Known limitations

- **Does not generalize zero-shot to external sites.** See above — this is
  the central, measured finding of this project, not a footnote.
- **No "normal tissue" concept.** BreaKHis has no normal class; the model
  cannot be expected to recognize normal tissue until trained on examples of it.
- **YOLO fails to detect a region on the large majority of out-of-distribution
  images** (BACH: no detection on 88%+ of images at default confidence).
- **Stain normalization helps but is not a fix**: accuracy improvement is
  highly sensitive to the reference image chosen (±8pp swing across a
  10-reference sweep) — the fragility is itself part of the finding.
- **Fine-tuning on external data is unresolved.** Improves held-out accuracy
  but (a) BACH patient IDs aren't fully recoverable so a leakage risk can't
  be ruled out, and (b) naive fine-tuning caused catastrophic forgetting on
  BreaKHis (94.71%→64.45%) without careful, forgetting-floor-constrained
  checkpoint selection.
- **This is a research prototype, not a validated clinical tool.** No
  pathologist has clinically signed off on model outputs. Do not use for
  actual diagnosis.

## Repo structure

```
stromasense-web/
├── README.md                          this file
├── ROADMAP.md                         build plan and current status
├── requirements.txt
├── app.py                             Streamlit entry point — Home page
├── pages/
│   ├── 1_Try_It.py                    upload/gallery + live pipeline + explainability
│   ├── 2_How_It_Works.py              architecture walkthrough
│   ├── 3_Results_and_Limitations.py   metrics, honest generalization gap
│   └── 4_Roadmap_and_Team.py          real-world path, team, contact
├── src/
│   ├── config.py                      paths, thresholds, class names, contact — EDIT THIS FIRST
│   ├── model_loader.py                loads YOLO / ResNet50 / ViT / XGBoost, HF Hub download
│   ├── inference.py                   full pipeline orchestration (now also times each run)
│   ├── explainability.py              GradCAM + ViT attention rollout
│   ├── model_card.py                  generates a downloadable Model Card from metrics.json
│   ├── components.py                  shared UI (disclaimer banner, etc.)
│   └── utils.py                       image transforms, overlay drawing
├── data/
│   └── metrics.json                   single source of truth for all displayed numbers
├── scripts/
│   └── test_pipeline_standalone.py    Day-1 sanity check, run before any UI work
└── assets/
    ├── sample_images/                 curated gallery images + manifest.json (ground truth)
    └── diagrams/                      architecture diagram(s)
```

## Getting started

```bash
git clone <this-repo-url>
cd stromasense-web
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Edit `src/config.py`: set your Hugging Face Hub model repo ID (or local
   weight paths), `CONTACT_EMAIL` and `TEAM_NAME` (used on the Roadmap page
   and in the downloadable Model Card), and confirm class names / thresholds.
2. Populate `assets/sample_images/` and `assets/sample_images/manifest.json`
   with real curated images and their ground-truth labels.
3. Fill `data/metrics.json` with your actual numbers if they differ from the
   template values — the Results page, the Model Card download, and the
   Home page metrics all read from this one file, so this is the only
   place you need to update.
4. Run locally:

```bash
streamlit run app.py
```

### Deploying

This app is designed for **Hugging Face Spaces** (Streamlit SDK). Model
weights are pulled from a Hugging Face Hub model repo at startup rather than
committed to this repo — see `src/model_loader.py`.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the build plan, and the in-app "Roadmap &
Team" page for the intended real-world path (local clinical data
partnerships, magnification/scale-matched preprocessing, and the
forgetting-floor-constrained fine-tuning work still in progress).

## Datasets & citations

- **BreaKHis**: Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L. *A
  Dataset for Breast Cancer Histopathological Image Classification*, IEEE
  Transactions on Biomedical Engineering (TBME), 63(7):1455-1462, 2016.
- **BACH**: Aresta, G. et al. *BACH: Grand Challenge on Breast Cancer
  Histology Images*, Medical Image Analysis, 2019 (ICIAR 2018 Challenge).
- **This work**: _add your IEEE Access citation here._

## License

_Choose and add a license (MIT/Apache-2.0 are common for this kind of
research code). Note that BreaKHis and BACH each carry their own
non-commercial research usage terms independent of this repo's license —
don't commit the raw datasets here._
