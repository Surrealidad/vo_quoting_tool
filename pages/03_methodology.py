"""
pages/03_methodology.py — Methodology

Three sections:
1. Why filecount is a poor predictor — the business argument
2. SHAP feature importance — what the model actually learned
3. Full training code — transparency
"""

import streamlit as st
from pathlib import Path
from utils import load_metadata

MODEL_DIR = Path(__file__).parent.parent / "models"
TRAIN_PY  = Path(__file__).parent.parent / "models" / "train.py"

metadata = load_metadata()
metrics  = metadata["metrics"]

# ─── PAGE HEADER ───────────────────────────────────────────────────────────────

st.title("🔬 Methodology")
st.caption(
    "How the model was built, what it learned, and why it outperforms "
    "the traditional approach. Full training code included."
)
st.divider()

# ─── SECTION 1: THE BUSINESS ARGUMENT ─────────────────────────────────────────

st.subheader("1 — Why filecount is a poor predictor")

st.markdown("""
The traditional quoting method in VO localization multiplies **filecount × a fixed
rate per language and time restriction**. This is auditable and fast — but it rests
on a flawed assumption: that every file represents the same amount of recording work.

It does not.

**Words per file varies significantly by session type:**

| Time Restriction | Avg. Words per File |
|---|---|
| LipSync | ~4.5 |
| SoundSync | ~5.5 |
| HTR | ~6.0 |
| STR | ~7.0 |
| NoTR | ~8.0 |

A LipSync file and a NoTR file cost the same in the traditional model — but the NoTR
file contains almost twice the dialogue. When studios submit large NoTR sessions with
dense scripts, the traditional method **underestimates**. When they submit LipSync
sessions with many short, precisely timed lines, it **overestimates**.

Beyond words per file, the traditional method ignores:

- **Minimum fees** — actors charge a minimum regardless of scope. Thin per-actor
  scopes (high spread) inflate cost relative to wordcount.
- **Cast size relative to scope** — 80 actors covering 5,000 words is a very
  different session to 10 actors covering the same material.

These are not edge cases. They are systematic, predictable, and correctable.
""")

st.divider()

# ─── SECTION 2: SHAP ───────────────────────────────────────────────────────────

st.subheader("2 — What the model learned: SHAP feature importance")

st.markdown("""
SHAP (SHapley Additive exPlanations) decomposes each prediction into contributions
from individual features — showing not just *which* features matter, but *how* and
*in which direction* they push the predicted cost.

**How to read the chart below:**
- Each row is one feature, ordered by importance (most impactful at the top)
- Each dot is one session from the test set
- **X-axis**: SHAP value — how much that feature pushed the prediction up (right) or down (left)
- **Colour**: the feature's value for that session (red = high, blue = low)
""")

shap_path = MODEL_DIR / "shap_summary.png"
if shap_path.exists():
    st.image(str(shap_path), use_container_width=True)
else:
    st.warning("SHAP plot not found — run models/train.py to generate it.")

st.markdown("""
**Reading the key features:**

**Wordcount** (top) — red dots (large scripts) push cost strongly upward.
The most important raw driver, as expected.

**Words_per_File** (second) — this is the insight the traditional method misses.
Red dots (dense files, many words per file) push cost *up*. Blue dots (sparse files,
few words per file) push cost *down*. The model has learned that filecount alone
is an unreliable workload proxy.

**TR_Score** — the time restriction ordinal. High TR (LipSync = 4) pushes cost up.
Confirms the model correctly learned the multiplier hierarchy.

**Language** — high-rate languages (Japanese, Korean) push cost up; low-rate
(Polish, LATAM) push it down. The model recovered the language rate table from
data alone, without being given the actual rates.

**Words_per_Actor** — thin per-actor scope (low values, blue) is associated with
minimum fee exposure, which pushes cost up. The model detected this non-linearity.

**Spread_Score** — confirms the spread premium is being learned, though it ranks
lower because wordcount and WPF carry most of the signal.
""")

st.divider()

# ─── SECTION 3: TRAINING CODE ──────────────────────────────────────────────────

st.subheader("3 — Training code")

st.markdown("""
The model was trained locally and saved as `model.pkl`. This app loads the
pre-trained model at startup — no training happens here. Below is the full
training script for transparency.

Key decisions worth noting:
- **XGBoost** was chosen over linear regression because the cost function is
  non-linear (minimum fees, TR multipliers interacting with spread)
- **OrdinalEncoder** for language and TR — not one-hot — because these categories
  have meaningful ordinality (TR_Score encodes cost hierarchy explicitly)
- **20% test split** + **5-fold CV** to avoid overfitting on a modest dataset
- **Noise was deliberately introduced** in data generation (~8% lognormal std dev)
  so the model cannot achieve unrealistically perfect accuracy
""")

with st.expander("View full training code (train.py)", expanded=False):
    if TRAIN_PY.exists():
        with open(TRAIN_PY, "r") as f:
            st.code(f.read(), language="python")
    else:
        st.warning("train.py not found.")
