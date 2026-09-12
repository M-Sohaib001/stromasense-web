"""
Generates a Model Card (Mitchell et al., 2019 / Hugging Face format) from
data/metrics.json plus fixed narrative content — so the card is always in
sync with whatever's actually shown on the Results page, not a separately
maintained document that drifts out of date.

Reused, not rewritten: every claim in this card is content already
established elsewhere in this repo — README.md's Known Limitations section
and the Results & Limitations page. This file just reformats it into a
standalone, downloadable, standard document.
"""

from datetime import date

from src.config import CONTACT_EMAIL, TEAM_NAME, CLASS_NAMES


def generate_model_card_markdown(metrics: dict) -> str:
    h = metrics["headline"]
    per_class = metrics["bach_per_class_accuracy"]
    yolo = metrics["yolo_detection_rate"]
    sn = metrics["stain_normalization_sweep"]
    ft = metrics["fine_tuning"]

    return f"""# Model Card: StromaSense (Patient-Split)

**Generated:** {date.today().isoformat()}
**Team:** {TEAM_NAME}
**Contact:** {CONTACT_EMAIL}

## Model description

A cascaded pipeline (YOLO region gate → ResNet50 + ViT-Base fused features →
XGBoost) classifying breast tissue histopathology crops as **{" / ".join(CLASS_NAMES)}**,
with Grad-CAM and attention-rollout explainability on every prediction.

## Intended use

- Academic research and hackathon demonstration of a fused CNN+Transformer
  histopathology classification approach.
- Illustrating, with measured evidence, the cross-site generalization
  problem in computational pathology and a roadmap toward addressing it.

## Out-of-scope / prohibited use

- **Clinical diagnosis or any real patient care decision, in any capacity.**
- Use as a sole or primary basis for triage without qualified pathologist
  review.
- Use on inputs outside histopathology imagery (this is not a general
  medical image classifier).
- Any deployment implying regulatory clearance or clinical validation —
  neither has occurred.

## Training data

**BreaKHis** (Spanhol et al., 2016): 7,909 microscopy images from 82
patients, 40x-400x magnification, benign/malignant labels. Patient-level
train/test split (see "Known limitations" below for why this matters).

## Evaluation data

- **In-distribution:** BreaKHis held-out patient-split test set.
- **Out-of-distribution:** BACH / ICIAR 2018 Challenge (Aresta et al.,
  2019) — 400 images, 4 classes, different lab/scanner/staining protocol,
  never used in training.

## Metrics

| | BreaKHis (patient-split, in-distribution) | BACH (zero-shot, external) |
|---|---|---|
| Accuracy | {h['breakhis_patient_split']['accuracy']:.1%} | {h['bach_zero_shot']['accuracy']:.1%} |
| MCC | {h['breakhis_patient_split']['mcc']:.2f} | {h['bach_zero_shot']['mcc']:.2f} |
| ROC-AUC | {h['breakhis_patient_split']['roc_auc']:.3f} | {h['bach_zero_shot']['roc_auc']:.3f} |

> {h['breakhis_patient_split'].get('caveat', '')}

**BACH per-class accuracy** (classifier only, bypassing the YOLO gate):
Normal {per_class['normal']:.0%} · Benign {per_class['benign']:.0%} ·
In-situ {per_class['in_situ']:.0%} · Invasive {per_class['invasive']:.0%}

## Known limitations

1. **Does not generalize zero-shot to external sites.** This is the central
   measured finding of this project. Sensitivity (catching malignant-type
   tissue) transfers across datasets; specificity does not.
2. **No "normal tissue" concept** — absent from training data entirely.
3. **YOLO region-detection fails on {1 - yolo['bach']:.0%} of BACH images**
   at the default confidence threshold.
4. **Stain normalization helps but is fragile**: a single reference gave
   {sn['single_reference_result']:.1%} accuracy; a 10-reference sweep ranged
   {sn['min_across_10_refs']:.0%}-{sn['max_across_10_refs']:.0%} — the
   method has no single canonical target to converge on.
5. **Fine-tuning on external data status: {ft['status']}** Leakage risk
   ({ft['leakage_risk']}) and catastrophic forgetting
   ({ft['breakhis_accuracy_before_finetune']:.1%} →
   {ft['breakhis_accuracy_after_naive_finetune']:.1%} on BreaKHis under
   naive fine-tuning) are both open issues, not resolved ones.

## Ethical considerations

No clinical or pathologist sign-off has been obtained for any model output.
Explainability outputs (Grad-CAM, attention rollout) show what the model
attended to, not whether that attention is clinically meaningful — they
should not be read as a clinical justification. Any future deployment
requires local clinical data, ethics-committee-approved validation, and a
named clinical collaborator before any claim of diagnostic utility.

## Citations

- Spanhol, F. et al. *A Dataset for Breast Cancer Histopathological Image
  Classification*, IEEE TBME, 63(7):1455-1462, 2016.
- Aresta, G. et al. *BACH: Grand Challenge on Breast Cancer Histology
  Images*, Medical Image Analysis, 2019.
"""