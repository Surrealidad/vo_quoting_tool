"""
app.py — VO Session Cost Estimator
Entry point. Defines navigation between the three pages.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title = "VO Cost Estimator",
    page_icon  = "🎙️",
    layout     = "wide",
)

pages = [
    st.Page("pages/01_predictor.py",   title="Quote Predictor",   icon="🎙️"),
    st.Page("pages/02_accuracy.py",    title="Model Accuracy",    icon="📊"),
    st.Page("pages/03_methodology.py", title="Methodology",       icon="🔬"),
]

pg = st.navigation(pages)
pg.run()
