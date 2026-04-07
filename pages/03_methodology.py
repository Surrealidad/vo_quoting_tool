"""
pages/03_methodology.py — Methodology

Three sections:
1. Why the classic method has a structural ceiling — the business argument
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
    "How the model was built, what it learned, and where the classic "
    "approach reaches its limits. Full training code included."
)
st.divider()

# ─── SECTION 1: THE BUSINESS ARGUMENT ─────────────────────────────────────────

st.subheader("1 — Where the classic method reaches its limits")

st.markdown("""
The classic quoting method in VO localization multiplies **wordcount by a fixed
rate per language and time restriction type**. This is a reasonable starting point —
it accounts for script volume, language market rates, and the recording complexity
associated with each restriction type.

Its limitation is structural: the formula treats all sessions of the same wordcount
and TR type as equivalent, regardless of how the scope is distributed across the cast.

**Average words per file varies by session type:**

| Time Restriction | Avg. Words per File |
|---|---|
| LipSync | ~4.5 |
| SoundSync | ~5.5 |
| HTR | ~6.0 |
| STR | ~7.0 |
| NoTR | ~8.0 |

This variation matters less for a word-based approach than for a file-based one,
since wordcount is already the primary input. The classic method's main blind spot
is elsewhere: **minimum fee dynamics**.

When a large number of actors each cover a small portion of the script, each actor
triggers a minimum session fee regardless of their individual scope. The per-word
formula has no mechanism to detect this — a session with 120 actors covering 15,000
words produces the same estimate as one with 15 actors covering the same material,
despite the significant difference in actual cost.

The analysis confirms this pattern. The sessions where the classic method performs
worst are consistently NoTR sessions with large casts and high scope spread, where
minimum fees dominate the actual billing and the per-word rate is a poor predictor
of the final invoice.

Beyond minimum fees, the classic method also has no visibility into:

- **Words per actor** — a proxy for how thin each actor's individual scope is
- **Scope spread** — how unevenly distributed the material is across the cast
- **Characters per actor** — relevant where one actor covers several characters

These are the factors the model was trained to capture.
""")

st.divider()

# ─── SECTION 2: SHAP ───────────────────────────────────────────────────────────

st.subheader("2 — What the model learned: SHAP feature importance")

st.markdown("""
SHAP (SHapley Additive exPlanations) decomposes each prediction into contributions
from individual features — showing not just *which* features matter, but *how* and
*in which direction* they influence the predicted cost.

**How to read the chart below:**
- Each row is one feature, ordered by overall importance (most impactful at top)
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
**Key features and what they show:**

**Wordcount** — the primary driver, consistent with both the classic method and
domain knowledge. Large scripts (red) push cost up significantly.

**Words_per_File** — sessions with fewer words per file (blue) push cost up,
not down. This reflects the minimum fee effect: sparse files indicate thin
per-actor scope, which increases cost per word relative to a denser session.
The classic method cannot capture this relationship.

**TR_Score** — the ordinal encoding of time restriction type. Higher TR
(LipSync = 4) pushes cost up, confirming the model has correctly learned
the multiplier hierarchy.

**Language** — high-rate languages (Japanese, Korean) push cost up; low-rate
languages (Polish, LATAM) push it down. The model recovered the language rate
structure from data without being given the explicit rate table.

**Words_per_Actor** — thin per-actor scope (low values, blue) is associated
with minimum fee exposure and pushes predicted cost upward. This is the
interaction the classic method misses most in large-cast sessions.

**Spread_Score** — scope spread adds signal on top of words-per-actor,
though it ranks lower because the actor-level features carry most of the
minimum fee information.
""")

st.divider()

# ─── SECTION 3: TRAINING CODE ──────────────────────────────────────────────────

st.subheader("3 — Training code")

st.markdown(f"""
The model was trained locally and saved as `model.pkl`. This app loads the
pre-trained model at startup — no training happens at runtime.

Key decisions in the training pipeline:

- **XGBoost** was chosen over linear regression because the cost function is
  non-linear: minimum fees create a floor effect, and TR multipliers interact
  with spread in ways a linear model cannot represent cleanly.
- **OrdinalEncoder** for language and TR rather than one-hot encoding, because
  both have meaningful ordinality (TR_Score explicitly encodes cost hierarchy).
- **20% test split plus 5-fold cross-validation** to guard against overfitting
  on a dataset of roughly 1,000 training samples.
- **Noise was deliberately introduced** during data generation (lognormal, ~8%
  std dev) so the model cannot achieve unrealistically low error. The resulting
  test MAPE of {metrics['mape_model_test']:.1f}% reflects genuine prediction
  difficulty, not a clean synthetic signal.
""")

with st.expander("View full training code (train.py)", expanded=False):
    if TRAIN_PY.exists():
        with open(TRAIN_PY, "r") as f:
            st.code(f.read(), language="python")
    else:
        st.warning("train.py not found.")
