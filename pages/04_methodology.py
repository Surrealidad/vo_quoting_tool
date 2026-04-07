"""
pages/04_methodology.py — Methodology
Data generation, model design, SHAP analysis, training code.
"""

import streamlit as st
from pathlib import Path
from utils_v2 import load_metadata_s1, load_metadata_s2

MODEL_DIR = Path(__file__).parent.parent / "models"
TRAIN_S1  = Path(__file__).parent.parent / "models" / "train_stage1.py"
TRAIN_S2  = Path(__file__).parent.parent / "models" / "train_stage2.py"

m1 = load_metadata_s1()
m2 = load_metadata_s2()

st.title("🔬 Methodology")
st.caption(
    "How the dataset was constructed, how each model was trained, "
    "and what the models learned. Full training code included."
)
st.divider()

# ── SECTION 1: DATA GENERATION ────────────────────────────────────────────────
st.subheader("1 — Synthetic dataset")

st.markdown("""
Both models are trained on a synthetic dataset generated from production domain
knowledge rather than real vendor invoices. This approach was chosen because
actual billing data from localisation vendors is commercially sensitive, but the
cost structure of VO sessions is well understood and can be encoded directly.

**Generation approach:**

The dataset is built from the ground up at actor level (Stage 2), then aggregated
to session level (Stage 1). This ensures the two datasets are internally consistent —
Stage 1 session actuals are the exact sum of Stage 2 actor actuals.

Each actor has a persistent efficiency factor drawn once from
Normal(mean=1.05, std=0.30), representing their tendency to record faster or slower
than the standard 1,000 normalised words per hour. This factor is the main signal
the Stage 2 model is trained to learn from session history.

**Sources of realistic variance introduced:**

- **Script delivery variance** — actual recorded scope differs from the quoted scope
  by a factor drawn from Normal(1.05, 0.15), reflecting scripts that grow or shrink
  between quote and recording day
- **Retake sessions** — 25% of sessions include unplanned retake time (10–40%
  additional hours), invisible to the classic formula
- **Billing discretisation** — hours are billed in whole units with a 15-minute
  grace period before rounding up, creating systematic step-function noise
- **VIP actors** — approximately 5% of actors command 2–3× the standard rate;
  Stage 1 classic uses a flat 1.10 buffer that cannot anticipate the exact VIP
  composition of any given session

**Dataset scale:**
- 988 Stage 1 session records across 32 projects, 11 languages
- 22,604 Stage 2 actor-session records
- 1,745 unique actors across 10 vendors
- Production dates spanning 2021–2027, including projected future sessions
""")

st.divider()

# ── SECTION 2: CLASSIC BASELINES ──────────────────────────────────────────────
st.subheader("2 — Classic baselines")

st.markdown(f"""
Each model is benchmarked against a classic formula that represents the
standard industry quoting approach.

**Stage 1 classic:**
Total estimated session hours × standard hourly rate × 1.10 flat buffer.
The 1.10 is intended to cover VIP rates and scope variance, but it is fixed
regardless of session characteristics. MAPE: **{m1['metrics']['mape_classic']:.1f}%**.

**Stage 2 classic:**
Per-actor estimated hours × individual actor rate (VIP-aware). This is more
precise than Stage 1 because it knows the actual cast and their rates, but
it still assumes every actor records at exactly 1,000 normalised words per hour.
It has no visibility into retakes or individual recording pace.
MAPE: **{m2['metrics']['mape_classic']:.1f}%**.

The gap between the two classic MAPEs reflects the value of cast information
alone, before any machine learning is applied.
""")

st.divider()

# ── SECTION 3: MODEL DESIGN ───────────────────────────────────────────────────
st.subheader("3 — Model design decisions")

tab1, tab2 = st.tabs(["Stage 1", "Stage 2"])

with tab1:
    st.markdown(f"""
**Algorithm:** XGBoost regressor. Linear regression was considered but the cost
function is non-linear — TR mix interactions and minimum fee exposure create
threshold effects that a linear model cannot represent cleanly.

**Target:** Absolute session cost in EUR.

**Key features:**
- Normalised wordcount and estimated hours (primary workload signal)
- TR mix percentages (Pct_LS, Pct_SS, Pct_HTR, Pct_STR) — capture
  restriction complexity beyond the flat rate multiplier
- Number of actors — drives minimum fee exposure
- Standard rate per hour — language × vendor modifier
- Is_Sequel — modest but consistent signal in the data

**Encoding:** OrdinalEncoder for Language and Vendor. Not one-hot,
because ordinal encoding preserves a meaningful rate hierarchy.

**Validation:** 80/20 train/test split + 5-fold cross-validation.
Test MAPE: {m1['metrics']['mape_model_test']:.1f}%. CV MAPE: {m1['metrics']['mape_model_cv']:.1f}%.
R²: {m1['metrics']['r2']:.3f}.
""")

with tab2:
    st.markdown(f"""
**Algorithm:** XGBoost regressor.

**Target:** Cost ratio (actual / classic forecast) rather than absolute cost.
This was a deliberate design choice — predicting the ratio focuses learning on
*how much the classic formula is wrong* rather than reconstructing cost from scratch.
The final prediction is: ratio_prediction × Cost_Forecast_S2.

**Key features:**
- Per-actor TR word counts (LS, SS, HTR, STR, NoTR)
- Normalised wordcount and estimated hours
- Cost per hour (includes VIP premium if applicable)
- Cost_Forecast_S2 — the classic estimate, used as a strong prior
- Actor_Hist_Ratio — actor's historical mean of (actual / forecast), computed
  from training sessions only to prevent target leakage
- Actor_Session_Count — number of sessions seen in training; low values
  indicate limited history and reduced confidence
- Actor_Hist_Std — variance in historical ratio; high values indicate
  unpredictable actors

**Leakage prevention:** Actor historical features are computed from the training
set only, then mapped to the test set. Actors not present in training receive
the global mean ratio ({m2['global_mean_ratio']:.3f}).

**Session-based split:** The train/test split is performed at session level
(all actors in a session stay in the same split) rather than at row level, to
prevent data leakage through shared session characteristics.

**Validation:** Test MAPE: {m2['metrics']['mape_model_test']:.1f}%.
CV MAPE: {m2['metrics']['mape_model_cv']:.1f}%. R²: {m2['metrics']['r2']:.3f}.
""")

st.divider()

# ── SECTION 4: SHAP ───────────────────────────────────────────────────────────
st.subheader("4 — Feature importance (SHAP)")
st.markdown("""
SHAP (SHapley Additive exPlanations) decomposes each prediction into contributions
from individual features. Each dot represents one observation from the test set.
The x-axis shows the SHAP value — how much that feature pushed the prediction
up (right) or down (left). Colour indicates the feature value (red = high, blue = low).
""")

tab3, tab4 = st.tabs(["Stage 1", "Stage 2"])
with tab3:
    p = MODEL_DIR / "shap_s1.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage1.py to generate this plot.")
    st.markdown("""
**Stage 1 SHAP — what to observe:**

`Estimated_Hours` and `Normalized_Wordcount` dominate, as expected — these are
the primary workload signals. `Standard_Rate_per_Hour` captures the language and
vendor rate hierarchy. The TR mix percentages (`Pct_LS`, `Pct_STR`) add secondary
signal: high LipSync proportion pushes cost up relative to the formula's expectation,
reflecting the non-linear minimum fee exposure the classic method misses.
""")

with tab4:
    p = MODEL_DIR / "shap_s2.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage2.py to generate this plot.")
    st.markdown("""
**Stage 2 SHAP — what to observe:**

`Cost_Forecast_S2` ranks highest — the model uses the classic forecast as its
strongest prior and predicts a ratio adjustment around it. `Actor_Hist_Ratio`
should appear as the second most important feature, confirming that individual
actor efficiency history is the primary signal the model adds beyond the classic.
`Cost_per_Hour` captures VIP rate variation. `Actor_Session_Count` acts as a
confidence weight — actors with more history receive more precise adjustments.
""")

st.divider()

# ── SECTION 5: TRAINING CODE ──────────────────────────────────────────────────
st.subheader("5 — Training code")
st.markdown(
    "Both models are trained locally and saved as `.pkl` files. "
    "The app loads pre-trained models at startup — no training happens at runtime."
)

tab5, tab6 = st.tabs(["Stage 1 — train_stage1.py", "Stage 2 — train_stage2.py"])
with tab5:
    with st.expander("View full training code", expanded=False):
        if TRAIN_S1.exists():
            st.code(TRAIN_S1.read_text(), language="python")
        else:
            st.warning("train_stage1.py not found.")

with tab6:
    with st.expander("View full training code", expanded=False):
        if TRAIN_S2.exists():
            st.code(TRAIN_S2.read_text(), language="python")
        else:
            st.warning("train_stage2.py not found.")
