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
knowledge rather than real vendor invoices. Actual billing data from localisation
vendors is commercially sensitive, but the cost structure of VO sessions is well
understood and can be encoded directly.

**Generation approach:**

The dataset is built at actor level (Stage 2), then aggregated to session level
(Stage 1). This ensures internal consistency — Stage 1 session actuals are derived
from Stage 2 actor actuals, so the two datasets are always aligned.

**Cost forecast (classic baseline):**

The classic forecast at actor level is deterministic:
estimated hours × actor hourly rate, where hours are computed from normalised
wordcount (TR words weighted by multiplier: LipSync ×5, SoundSync ×4, HTR ×2.5,
STR ×2, NoTR ×1) and rounded up with a 15-minute grace period before billing
the next hour.

**Noise architecture — three layers:**

*Actor level (Stage 2):*
Each actor has a persistent `base_efficiency` factor drawn once from
Normal(1.0, 0.20), representing their tendency to record faster or slower than
standard. This is the learnable signal the Stage 2 model exploits across sessions.

On top of this, each session also applies two independent dice rolls:
- D4: 25% probability of ×0.7 (session ran short)
- D6: 17% probability of ×1.7 (significant overrun — retakes, direction issues)
- Residual Normal(1.0, 0.10): small additional noise

The dice are random per session and not learnable. The actor efficiency factor
is persistent and learnable. This combination produces realistic variance where
some of the error is systematic (actor-level) and some is irreducible noise.

*Project level (Stage 1):*
Each project has a persistent `project_efficiency` factor drawn once from
Normal(1.0, 0.25), representing project-wide characteristics: director style,
vendor relationship quality, script cleanliness. This factor is applied to all
Stage 1 actuals within the project, creating cross-session correlation that the
Stage 1 model can learn from. Sequels inherit a mild version of their parent
project's factor, with regression toward the mean.

**VIP actors:** approximately 5% of actors command 2–3× the standard hourly
rate. The Stage 1 classic uses a flat 1.10 buffer that cannot anticipate the
exact VIP composition of any session. The Stage 2 classic knows actual rates
but not efficiency patterns.

**Dataset scale:**
- Stage 1: ~650 past session records across 32 projects, 11 languages
- Stage 2: ~22,000 past actor-session records
- 1,745 unique actors across 10 vendors
- Production dates spanning 2021–2027
""")

st.divider()

# ── SECTION 2: CLASSIC BASELINES ──────────────────────────────────────────────
st.subheader("2 — Classic baselines")

st.markdown(f"""
Each model is benchmarked against a classic formula representing the standard
industry quoting approach.

**Stage 1 classic:**
Total estimated session hours × standard hourly rate × 1.10 flat buffer.
The 1.10 buffer is intended to cover VIP rates and scope variance, but it is
fixed regardless of session or project characteristics.
MAPE: **{m1['metrics']['mape_classic']:.1f}%**.

**Stage 2 classic:**
Per-actor estimated hours × individual actor rate (VIP-aware). More precise
than Stage 1 because it knows the actual cast and their rates, but it assumes
every actor records at exactly the standard pace. It has no visibility into
individual efficiency patterns or session-level randomness.
MAPE: **{m2['metrics']['mape_classic']:.1f}%**.

The gap between Stage 1 and Stage 2 classic MAPEs reflects the value of cast
information alone, before any machine learning is applied.
""")

st.divider()

# ── SECTION 3: MODEL DESIGN ───────────────────────────────────────────────────
st.subheader("3 — Model design decisions")

tab1, tab2 = st.tabs(["Stage 1", "Stage 2"])

with tab1:
    st.markdown(f"""
**Algorithm:** XGBoost regressor. The cost function is non-linear — project
efficiency creates cross-session dependencies that a linear model cannot
represent cleanly.

**Target:** Absolute session cost in EUR.

**Key features:**
- Raw TR word counts per type (LS, SS, HTR, STR, NoTR) — the model learns
  the TR multiplier hierarchy from data rather than having it pre-computed
- Number of actors and estimated hours — primary workload signals
- Standard rate per hour — captures language and vendor rate structure
- Language and Vendor (ordinal encoded)
- Is_Sequel — projects inheriting a parent project's efficiency factor

**Why raw TR words instead of percentages:**
Using the five raw word columns gives the model both the shape (TR mix) and
the absolute volume of each restriction type, without pre-computing the
normalised wordcount. The model can discover the weighting relationships itself,
which is a more honest test of what it actually learns.

**Validation:** 80/20 train/test split + 5-fold cross-validation.
Test MAPE: **{m1['metrics']['mape_model_test']:.1f}%**. CV MAPE: {m1['metrics']['mape_model_cv']:.1f}%. R²: {m1['metrics']['r2']:.3f}.
Improvement over classic: **{m1['metrics']['improvement_pp']:.1f} pp**.
""")

with tab2:
    st.markdown(f"""
**Algorithm:** XGBoost regressor.

**Target:** Cost ratio (Cost_Actuals / Cost_Forecast) rather than absolute cost.
The model predicts how much the classic forecast is wrong, then the final
prediction is: predicted_ratio × Cost_Forecast_S2. This focuses learning on
the systematic deviation from the baseline rather than reconstructing cost
from scratch.

**Key features:**
- Per-actor TR word counts (LS, SS, HTR, STR, NoTR)
- Total_Words (raw sum, unweighted) — volume signal without pre-computed weights
- Estimated hours and cost per hour
- Cost_Forecast_S2 — the classic estimate used as a strong prior
- Is_VIP — rate multiplier signal
- Actor_Hist_Ratio — actor's historical mean of (actual / forecast), computed
  from training sessions only to prevent target leakage. This is the primary
  signal the model uses beyond the classic.
- Actor_Session_Count — confidence weight; actors with more history receive
  more precise adjustments
- Actor_Hist_Std — uncertainty signal; high variance actors are less predictable

**Leakage prevention:** actor historical features are computed from the training
set only, then mapped to the test set. Actors absent from training receive the
global mean ratio ({m2['global_mean_ratio']:.3f}).

**Session-based split:** the train/test split is performed at session level so
all actors in a session stay in the same partition, preventing leakage through
shared session characteristics.

**Validation:** Test MAPE: **{m2['metrics']['mape_model_test']:.1f}%**.
CV MAPE: {m2['metrics']['mape_model_cv']:.1f}%. R²: {m2['metrics']['r2']:.3f}.
Improvement over classic: **{m2['metrics']['improvement_pp']:.1f} pp**.
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
**Stage 1 SHAP — reading the chart:**

`Estimated_Hours` is the dominant feature by a large margin. High values (red)
push the prediction strongly upward, as expected — more hours in the booth means
higher cost. The wide spread of dots confirms this is the single most important
driver the model learned.

`Standard_Rate_per_Hour` ranks second, capturing the combined effect of language
market rate and vendor modifier. High rates (expensive languages like Japanese
or Korean) push cost up; low rates (Polish, LATAM) pull it down.

`LS_Words_Total` ranks third and shows a consistent rightward push — sessions
with LipSync scope are systematically more expensive relative to their estimated
hours, which the classic flat-rate formula does not fully capture.

`Number_of_Actors` adds a modest but consistent signal, reflecting the
interaction between cast size and session cost that goes beyond simple volume.

The remaining TR word columns (`HTR`, `STR`, `SS`, `NoTR`) contribute smaller
adjustments. `NoTR_Words_Total` sits at the bottom — once hours and rate are
known, additional wild scope adds relatively little incremental signal.

`Is_Sequel` has minimal SHAP impact, confirming it is a weak but present
signal — sequel projects tend to run closer to their parent's efficiency pattern,
but the effect is small compared to the primary workload features.
""")

with tab4:
    p = MODEL_DIR / "shap_s2.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Run models/train_stage2.py to generate this plot.")
    st.markdown("""
**Stage 2 SHAP — reading the chart:**

`Actor_Hist_Ratio` is the dominant feature by a very large margin — the chart
shows it accounting for the vast majority of the model's adjustment range
(approximately −0.6 to +1.0 on the ratio scale). Actors with a high historical
ratio (red — consistently over-delivering relative to forecast) receive strong
upward adjustments; fast actors (blue) receive downward adjustments. This
confirms that persistent actor efficiency is the primary signal the Stage 2
model exploits beyond the classic formula.

`Actor_Hist_Std` ranks second — actors with high variance in their historical
ratio (unpredictable performers) push the prediction upward slightly, reflecting
the model's learned caution around uncertain actors.

The TR word columns (`Total_Words`, `NoTR_Words`, `SS_Words`, `HTR_Words`,
`STR_Words`, `LS_Words`) contribute secondary adjustments, all clustered near
zero — the model uses them for fine-tuning but they are not the primary
decision drivers at actor level.

`Cost_Forecast_S2` ranks eighth rather than first, which is expected given
the model predicts a *ratio* rather than an absolute cost — the forecast is
already baked into the target definition, so the model focuses on what
deviates from it.

`Actor_Session_Count`, `Cost_per_Hour`, `Is_VIP`, `Language`, and `Vendor`
contribute minimal SHAP values, indicating that once actor history and scope
are accounted for, these features add little further precision.
""")

st.divider()

# ── SECTION 5: TRAINING CODE ──────────────────────────────────────────────────
st.subheader("5 — Training code")
st.markdown(
    "Both models are trained locally and saved as `.pkl` files. "
    "The app loads pre-trained models at startup — no training happens at runtime. "
    "Run `python models/train_stage1.py` then `python models/train_stage2.py` "
    "from the project root to regenerate all model artefacts and plots."
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

