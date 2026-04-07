"""
app.py — VO Session Cost Estimator (Two-Stage)
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title = "VO Cost Estimator",
    page_icon  = "🎙️",
    layout     = "wide",
)

pages = [
    st.Page("pages/00_home.py",        title="Overview",          icon="🏠"),
    st.Page("pages/01_stage1.py",      title="Stage 1 — Session", icon="📋"),
    st.Page("pages/02_stage2.py",      title="Stage 2 — Actors",  icon="🎙️"),
    st.Page("pages/03_accuracy.py",    title="Model Accuracy",    icon="📊"),
    st.Page("pages/04_methodology.py", title="Methodology",       icon="🔬"),
]

pg = st.navigation(pages)
pg.run()
