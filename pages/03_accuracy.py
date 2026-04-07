"""
pages/03_accuracy.py — Model Accuracy
All diagnostic charts for both stages side by side.
"""

import streamlit as st
from pathlib import Path
from utils_v2 import load_metadata_s1, load_metadata_s2

MODEL_DIR = Path(__file__).parent.parent / "models"

m1 = load_metadata_s1()['metrics']
m2 = load_metadata_s2()['metrics']

st.title("📊 Model Accuracy")
st.caption(
    "Performance of both models against the classic flat-rate baseline, "
    "evaluated on a held-out 20% test set."
)
st.divider()

# ── TOP KPIs ─────────────────────────────────────────────────────────────────
st.subheader("Summary metrics")

cols = st.columns(4)
cols[0].metric("Stage 1 — Classic MAPE",   f"{m1['mape_classic']:.1f}%")
cols[1].metric("Stage 1 — Model MAPE",     f"{m1['mape_model_test']:.1f}%",
               delta=f"-{m1['improvement_pp']:.1f} pp", delta_color="inverse")
cols[2].metric("Stage 2 — Classic MAPE",   f"{m2['mape_classic']:.1f}%")
cols[3].metric("Stage 2 — Model MAPE",     f"{m2['mape_model_test']:.1f}%",
               delta=f"-{m2['improvement_pp']:.1f} pp", delta_color="inverse")

c2 = st.columns(4)
c2[0].metric("Stage 1 — R²", f"{m1['r2']:.3f}")
c2[1].metric("Stage 1 — CV MAPE", f"{m1['mape_model_cv']:.1f}%",
             help="5-fold cross-validated MAPE on training set.")
c2[2].metric("Stage 2 — R²", f"{m2['r2']:.3f}")
c2[3].metric("Stage 2 — CV MAPE", f"{m2['mape_model_cv']:.1f}%")

st.divider()

# ── PREDICTED VS ACTUAL ───────────────────────────────────────────────────────
st.subheader("Predicted vs actual cost")
st.caption(
    "Each dot is one session (Stage 1) or actor-session (Stage 2) from the test set. "
    "The tighter the cloud around the diagonal, the lower the MAPE."
)

tab1, tab2 = st.tabs(["Stage 1", "Stage 2"])
with tab1:
    p = MODEL_DIR / "pred_actual_s1.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage1.py to generate this plot.")

with tab2:
    p = MODEL_DIR / "pred_actual_s2.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage2.py to generate this plot.")

st.divider()

# ── MAPE AND EUR ERROR BY ARCHETYPE ──────────────────────────────────────────
st.subheader("Quoting error by project size")
st.caption(
    "Left: MAPE (%) by project archetype. "
    "Right: Mean absolute error in EUR/USD — the financial impact of the accuracy gap."
)

tab3, tab4 = st.tabs(["Stage 1", "Stage 2"])
with tab3:
    p = MODEL_DIR / "arch_breakdown_s1.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage1.py to generate this plot.")

with tab4:
    p = MODEL_DIR / "arch_breakdown_s2.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage2.py to generate this plot.")

st.divider()

# ── BUDGET ACCURACY ───────────────────────────────────────────────────────────
st.subheader("Project-level budget accuracy")
st.caption(
    "Errors at session level partially cancel when aggregated to project level. "
    "Budget accuracy = 1 − |forecast − actual| / actual, expressed as a percentage. "
    "Billing in USD (cost + 10% markup). Higher is better."
)

tab5, tab6 = st.tabs(["Stage 1", "Stage 2"])
with tab5:
    p = MODEL_DIR / "budget_accuracy_s1.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage1.py to generate this plot.")

with tab6:
    p = MODEL_DIR / "budget_accuracy_s2.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage2.py to generate this plot.")

st.divider()

# ── PER-PROJECT IMPROVEMENT ───────────────────────────────────────────────────
st.subheader("Per-project budget deviation improvement")
st.caption(
    "Each dot is one project. Position on the x-axis shows total actual billing; "
    "position on the y-axis shows how much the model reduced the budget deviation "
    "compared to the classic forecast (in USD thousands). "
    "Dot size scales with total project billing. "
    "Dots above the zero line represent projects where the model was more accurate."
)

tab7, tab8 = st.tabs(["Stage 1", "Stage 2"])
with tab7:
    p = MODEL_DIR / "project_improvement_s1.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage1.py to generate this plot.")

with tab8:
    p = MODEL_DIR / "project_improvement_s2.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage2.py to generate this plot.")

st.divider()

# ── PLAIN ENGLISH SUMMARY ─────────────────────────────────────────────────────
st.subheader("What the numbers mean")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"""
**Stage 1** operates before casting. The classic method applies a standard hourly
rate with a 10% flat buffer, adjusted for language and vendor. It has no visibility
into TR mix interactions or the non-linear relationship between actor count and
minimum fee exposure.

The model reduces average error from **{m1['mape_classic']:.1f}% to {m1['mape_model_test']:.1f}%**
at session level. At project level this translates to budget accuracy improvements
of up to 11 percentage points on Mega projects, where the compounding effect
of many sessions amplifies the formula's structural limitations.
""")

with col_b:
    st.markdown(f"""
**Stage 2** operates once the cast is confirmed. The classic method at this stage
knows each actor's rate — including VIP premiums — but still assumes everyone
records at the standard pace of 1,000 normalised words per hour.

The model improves on this by learning individual actor efficiency patterns from
session history. It predicts a ratio (actual / forecast) per actor rather than
an absolute cost, then scales the classic estimate accordingly. Average error
falls from **{m2['mape_classic']:.1f}% to {m2['mape_model_test']:.1f}%**, with
the largest gains on AAA and Mega projects where actor histories are most complete.
""")
