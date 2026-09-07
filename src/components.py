"""Shared UI components — import disclaimer_banner() at the top of every page."""

import streamlit as st
from src.config import DISCLAIMER_TEXT


def disclaimer_banner():
    """
    Call this once near the top of every page (app.py and every file in
    pages/). It's easy to add this to one page and forget the rest —
    check this explicitly during Day 4/5 of ROADMAP.md.
    """
    st.warning(DISCLAIMER_TEXT)


def dataset_attribution_footer():
    """Call once near the bottom of any page that shows results or metrics."""
    st.caption(
        "Trained on BreaKHis (Spanhol et al., 2016). Externally validated on "
        "BACH / ICIAR 2018 Challenge (Aresta et al., 2019). Both datasets "
        "used under their respective non-commercial research terms."
    )
