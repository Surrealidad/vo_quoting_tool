"""
pages/00_home.py — Overview
Explains the two-stage forecasting approach and when to use each.
"""

import streamlit as st
from utils_v2 import load_metadata_s1, load_metadata_s2

st.title("🎙️ VO Session Cost Estimator")
st.caption(
    "A two-stage predictive quoting system for voice-over localisation sessions, "
    "built on synthetic data generated from production domain knowledge."
)
st.divider()

# ── THE PROBLEM ───────────────────────────────────────────────────────────────
st.subheader("The quoting problem")
st.markdown("""
Estimating the cost of a VO recording session involves several sources of
uncertainty that compound at different stages of production planning:

- The script scope (word count, time restriction mix) may change between
  initial quote and recording day
- Session costs are sensitive to cast composition — actor fees, minimum fee
  exposure, and individual recording pace all affect the final invoice
- No flat-rate formula accounts for all of these simultaneously

The standard industry approach — applying a fixed per-word or per-hour rate
adjusted for language and restriction type — produces estimates that are on
average **13–14% off** the actual invoice at session level. The gap is
systematic and predictable, not random.

This tool presents two machine learning models that reduce that error by
learning patterns from historical session data.
""")

st.divider()

# ── TWO STAGES ────────────────────────────────────────────────────────────────
st.subheader("Two forecasting stages")
st.caption(
    "Each stage corresponds to a point in the production timeline "
    "where different information is available."
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Stage 1 — Session estimate")
    st.markdown("""
    **When:** Before casting. You have the script breakdown but not the actor list.

    **Inputs required:**
    - Language and vendor
    - Word count by time restriction type
    - Number of actors (estimated)
    - Whether this is a sequel

    **What the model learns that the classic formula misses:**
    The interaction between TR mix, actor count, and session cost is
    non-linear. Sessions with a high proportion of LipSync scope and many
    actors create disproportionate minimum fee exposure that a flat rate
    does not capture.

    **Typical accuracy:** Classic ~14% error → Model ~12% error at session
    level, improving to ~99% budget accuracy on Mega projects at project level.
    """)

with col2:
    st.markdown("#### Stage 2 — Actor estimate")
    st.markdown("""
    **When:** Cast is confirmed. You know which actors are recording and
    how much material each one covers.

    **Inputs required:**
    - Everything from Stage 1, plus
    - Per-actor word count by TR type
    - Actor names (for historical efficiency lookup)
    - VIP flag per actor

    **What the model learns that the classic formula misses:**
    Individual actors have consistent efficiency patterns — some reliably
    record faster than the standard rate, others slower. The model learns
    these patterns from session history and adjusts estimates accordingly.
    VIP rate premiums, which the Stage 1 flat buffer cannot anticipate
    precisely, are also captured at actor level.

    **Typical accuracy:** Classic ~12% error → Model ~10% error at actor
    level, improving to ~97% budget accuracy on Mega projects at project level.
    """)

st.divider()

# ── METRICS SUMMARY ───────────────────────────────────────────────────────────
st.subheader("Model performance at a glance")

try:
    m1 = load_metadata_s1()['metrics']
    m2 = load_metadata_s2()['metrics']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stage 1 — Classic MAPE", f"{m1['mape_classic']:.1f}%")
    c2.metric("Stage 1 — Model MAPE",   f"{m1['mape_model_test']:.1f}%",
              delta=f"-{m1['improvement_pp']:.1f} pp", delta_color="inverse")
    c3.metric("Stage 2 — Classic MAPE", f"{m2['mape_classic']:.1f}%")
    c4.metric("Stage 2 — Model MAPE",   f"{m2['mape_model_test']:.1f}%",
              delta=f"-{m2['improvement_pp']:.1f} pp", delta_color="inverse")
except Exception:
    st.info("Run the training scripts to populate model metrics.")

st.divider()

st.caption(
    "Dataset: 988 sessions across 32 projects, 11 languages, 1,745 actors. "
    "Stage 2: 22,604 actor-session records. "
    "Production dates: 2021–2027, including projected future sessions. "
    "All data is synthetic, generated from production domain knowledge."
)
