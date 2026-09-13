import json
import streamlit as st
import plotly.graph_objects as go

from src.components import disclaimer_banner, dataset_attribution_footer
from src.config import METRICS_PATH
from src.model_card import generate_model_card_markdown

METRIC_HELP = {
    "accuracy": "Percent of predictions that matched the true label. Simple, but "
                 "misleading alone on imbalanced data — read alongside MCC.",
    "mcc": "Matthews Correlation Coefficient: -1 to +1, where 0 is random guessing "
           "and 1 is perfect. Unlike accuracy, it stays meaningful even if one class "
           "is more common than the other — treat this as the more trustworthy "
           "single number of the two.",
    "roc_auc": "Probability the model ranks a random true-positive higher than a "
               "random true-negative, from 0.5 (random) to 1.0 (perfect separation).",
    "sensitivity": "Of all truly malignant cases, the fraction the model correctly caught. "
                   "High sensitivity = few missed cancers.",
    "specificity": "Of all truly non-malignant cases, the fraction the model correctly "
                   "labeled as such. Low specificity = lots of false alarms.",
}

st.set_page_config(page_title="Results & Limitations — StromaSense", page_icon="📊", layout="wide")
disclaimer_banner()
st.title("Results & Limitations")

st.markdown(
    """
This is the page we'd want a skeptical reviewer to read first. The honest
headline: **this model works well on the data it was trained on, and does
not yet generalize to a different lab/scanner/staining protocol.** We
measured that gap directly rather than assuming it away, and we think that
measurement — not the in-distribution accuracy number — is the actual
contribution of this project.
"""
)

try:
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
except FileNotFoundError:
    st.error("data/metrics.json not found — see README.md Getting Started.")
    st.stop()

st.markdown("### Headline metrics")
st.caption("Hover the ⓘ on any metric below if you're not sure what it means.")
h = metrics["headline"]
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{h['breakhis_patient_split']['accuracy']:.1%}",
            help=METRIC_HELP["accuracy"] + " (BreaKHis, patient-split, in-distribution, "
                 "pooled across train+val+test — see caveat below)")
col2.metric("MCC", f"{h['breakhis_patient_split']['mcc']:.2f}", help=METRIC_HELP["mcc"])
col3.metric("ROC-AUC", f"{h['breakhis_patient_split']['roc_auc']:.3f}", help=METRIC_HELP["roc_auc"])
if "caveat" in h["breakhis_patient_split"]:
    st.caption(f"⚠️ {h['breakhis_patient_split']['caveat']}")

st.markdown("**vs. external validation (BACH, zero-shot):**")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{h['bach_zero_shot']['accuracy']:.1%}",
            help=METRIC_HELP["accuracy"] + " (BACH, external, zero-shot)")
col2.metric("MCC", f"{h['bach_zero_shot']['mcc']:.2f}", help=METRIC_HELP["mcc"])
col3.metric("ROC-AUC", f"{h['bach_zero_shot']['roc_auc']:.3f}", help=METRIC_HELP["roc_auc"])

st.markdown("### Per-class breakdown on BACH")
st.caption(
    "Classifier-only (bypassing the YOLO gate), patient-split model, n=100 per class."
)
per_class = {k: v for k, v in metrics["bach_per_class_accuracy"].items() if isinstance(v, (int, float))}
fig = go.Figure(
    go.Bar(
        x=list(per_class.keys()),
        y=list(per_class.values()),
        text=[f"{v:.0%}" for v in per_class.values()],
        textposition="auto",
    )
)
fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"), yaxis_title="Accuracy")
st.plotly_chart(fig, width='stretch')

st.markdown(
    """
**Reading this correctly:** in-situ and invasive carcinoma both map to the
"malignant" concept the model was trained on, and both score well (86-88%).
Normal and benign both score poorly (22-23%) — and it's tempting to explain
this away as "the model never saw normal tissue," but **benign was one of
the two classes it *was* trained on.** The real pattern is that the model's
*sensitivity* (catching true malignant cases) generalizes across sites,
while its *specificity* does not — it over-predicts "malignant" on tissue
it wasn't trained to positively recognize, whether that's a genuinely
unseen concept (normal) or a seen concept with different visual
characteristics across datasets (benign).
"""
)
st.caption(f"ⓘ Sensitivity: {METRIC_HELP['sensitivity']}")
st.caption(f"ⓘ Specificity: {METRIC_HELP['specificity']}")

st.markdown("### YOLO detection rate")
yolo = metrics["yolo_detection_rate"]
st.markdown(
    f"- BACH: **{yolo['bach']:.0%}** of images get any detection above threshold\n"
    f"- BreaKHis held-out: **{yolo['breakhis_held_out']:.0%}**"
)

st.markdown("### Stain normalization — helps, but is not a fix")
sn = metrics["stain_normalization_sweep"]
st.markdown(
    f"""
On BACH (classification-only, no YOLO gate), Macenko normalization toward
a single reference moved accuracy from **{sn['no_normalization_baseline']:.1%}**
(no normalization) to **{sn['single_reference_result']:.1%}** — a real,
measured improvement, not a projection. But across a 10-reference sweep,
results ranged **{sn['min_across_10_refs']:.0%}–{sn['max_across_10_refs']:.0%}**
(mean {sn['mean_across_10_refs']:.0%}) — a spread almost as large as the
improvement itself. This isn't an implementation bug; stain normalization
methods remap toward whichever reference you choose, and there is no single
canonical "correct" target to converge on.
"""
)

st.markdown("### Fine-tuning on external data — status")
ft = metrics["fine_tuning"]
st.info(ft["status"])
st.markdown(
    f"""
- Held-out BACH accuracy after naive fine-tuning: {ft['held_out_bach_accuracy_naive']:.0%}
  (**leakage risk: {ft['leakage_risk']}** — BACH patient IDs aren't fully recoverable)
- BreaKHis accuracy before fine-tuning: {ft['breakhis_accuracy_before_finetune']:.1%}
- BreaKHis accuracy after **naive** fine-tuning: {ft['breakhis_accuracy_after_naive_finetune']:.1%}
  (catastrophic forgetting — a forgetting-floor-constrained fine-tuning
  approach is in progress specifically to fix this)
"""
)

st.markdown("### Model Card")
st.caption(
    "A standard-format summary of this model's intended use, limitations, and "
    "ethical considerations — everything above, reformatted as a single "
    "downloadable document. Generated live from the same numbers shown on "
    "this page, so it can't drift out of sync with them."
)
st.download_button(
    "⬇ Download Model Card (.md)",
    data=generate_model_card_markdown(metrics),
    file_name="stromasense_model_card.md",
    mime="text/markdown",
)

dataset_attribution_footer()