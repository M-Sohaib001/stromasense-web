import streamlit as st
from src.components import disclaimer_banner
from src.config import CONTACT_EMAIL

st.set_page_config(page_title="Roadmap & Team — StromaSense", page_icon="🗺️", layout="wide")
disclaimer_banner()
st.title("Roadmap & Team")

st.info(
    f"""
**Working in pathology, digital pathology infrastructure, or histopathology
research in Pakistan?** The single biggest thing missing from this project
is local clinical data. We're looking for a pathology department or research
group interested in a collaboration structured around mutual benefit — not
a one-way data request — under proper ethics-committee approval.

📧 **{CONTACT_EMAIL}** — tell us a bit about your setup and what a
useful first step would look like for you.
"""
)

st.markdown(
    f"""
### What would actually close the generalization gap

Not a bigger model — more, and more diverse, training data, plus fixes
targeted at the specific failure modes we measured:

- **Local clinical data.** The core fix. This means a real collaboration
  with a pathology department that has (or is building) digitized
  histopathology infrastructure — not just a data-sharing request, a
  partnership with mutual benefit (e.g., a free second-read/QA tool in
  exchange for de-identified data access, under proper ethics-committee
  approval).
- **Magnification/physical-scale-matched preprocessing.** BreaKHis and
  BACH have different pixel-to-micron scales; naive resizing to a fixed
  input size discards this. Rescaling to match a known physical field of
  view is computable exactly (both datasets publish their pixel scale) and
  testable as a clean ablation — unlike stain normalization, which has no
  single canonical target.
- **Forgetting-floor-constrained fine-tuning.** In progress: fine-tuning on
  external data while explicitly monitoring and refusing to save a
  checkpoint that drops BreaKHis performance below a floor, to avoid the
  catastrophic forgetting seen under naive fine-tuning.
- **A genuine "normal tissue" training signal**, since no amount of
  preprocessing recovers a class the model was never shown examples of.

### Team

_Add names, roles, and contact links here._

### Links

- GitHub: _this repo_
- Paper: _add IEEE Access link_
- Contact: {CONTACT_EMAIL} _(also add LinkedIn if you want it here)_
"""
)
