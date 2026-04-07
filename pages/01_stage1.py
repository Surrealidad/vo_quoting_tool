"""
pages/01_stage1.py — Stage 1 Session Predictor
Session-level estimate. No cast required.
"""

import streamlit as st
from utils_v2 import (
    predict_stage1, vendors_for_language,
    LANGUAGE_RATES, TR_LABELS, TR_KEYS, DEFAULT_FX,
)

st.title("📋 Stage 1 — Session Estimate")
st.caption(
    "Use this when you have a script breakdown but have not yet confirmed the cast. "
    "Inputs are at session level. The model improves on the classic flat-rate formula "
    "by learning TR mix interactions and session structure patterns."
)
st.divider()

# ── SESSION INPUTS ────────────────────────────────────────────────────────────
st.subheader("Session parameters")

col1, col2, col3 = st.columns(3)
with col1:
    language = st.selectbox(
        "Language",
        options=sorted(LANGUAGE_RATES.keys()),
        index=sorted(LANGUAGE_RATES.keys()).index("German"),
    )
with col2:
    vendor_opts = vendors_for_language(language)
    vendor = st.selectbox("Vendor", options=vendor_opts)
with col3:
    exchange_rate = st.number_input(
        "Exchange Rate (EUR → USD)", min_value=0.50, max_value=3.00,
        value=DEFAULT_FX, step=0.01, format="%.2f",
        help="Applied to billing totals. Default is the latest rate in the dataset.",
    )

col4, col5 = st.columns(2)
with col4:
    n_actors = st.number_input(
        "Estimated Number of Actors", min_value=1, max_value=200, value=10, step=1,
    )
with col5:
    is_sequel = st.checkbox("This is a sequel project", value=False)

st.divider()

# ── TR WORD COUNTS ────────────────────────────────────────────────────────────
st.subheader("Script scope by time restriction")
st.caption(
    "Enter the total word count for each restriction type in this session. "
    "Fields left at zero are excluded from the estimate."
)

tr_cols = st.columns(5)
tr_words = {}
for i, tr in enumerate(TR_KEYS):
    with tr_cols[i]:
        tr_words[tr] = st.number_input(
            TR_LABELS[tr], min_value=0, max_value=500_000,
            value=10000 if tr == 'NoTR' else 0, step=500,
            key=f"s1_tr_{tr}",
        )

total_words = sum(tr_words.values())
if total_words > 0:
    st.caption(
        f"Total: {total_words:,} words — "
        + " | ".join(
            f"{TR_LABELS[tr]}: {tr_words[tr]/total_words*100:.1f}%"
            for tr in TR_KEYS if tr_words[tr] > 0
        )
    )

st.divider()

# ── PREDICTION ────────────────────────────────────────────────────────────────
if total_words == 0:
    st.info("Enter word counts above to generate an estimate.")
    st.stop()

result = predict_stage1(
    language=language, vendor=vendor, is_sequel=is_sequel,
    tr_words=tr_words, n_actors=n_actors, exchange_rate=exchange_rate,
)

st.subheader("Session cost estimate")

r1, r2, r3, r4 = st.columns(4)
r1.metric(
    "Classic Forecast (EUR)",
    f"€{result['classic_cost']:,.0f}",
    help="Flat-rate formula: total hours × standard rate × 1.10 buffer.",
)
r2.metric(
    "Model Estimate (EUR)",
    f"€{result['model_cost']:,.0f}",
    delta=f"€{result['model_cost'] - result['classic_cost']:+,.0f} vs classic",
    delta_color="off",
    help="XGBoost model trained on session-level features.",
)
r3.metric(
    "Classic Billing (USD)",
    f"${result['classic_bill']:,.0f}",
    help=f"Cost + {int(result['classic_cost']*(0.10)):,} markup × {exchange_rate} FX.",
)
r4.metric(
    "Model Billing (USD)",
    f"${result['model_bill']:,.0f}",
)

st.divider()

# ── SESSION PROFILE ────────────────────────────────────────────────────────────
st.subheader("Derived session profile")
st.caption("Computed from your inputs — these are the features the model uses.")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Normalised Wordcount",
          f"{result['norm_wc']:,}",
          help="Words weighted by TR multiplier (LipSync = 5×, NoTR = 1×).")
p2.metric("Estimated Session Hours",
          f"{result['est_hours']:,}",
          help="Sum of per-actor billed hours at standard recording rate.")
p3.metric("Standard Rate (EUR/hr)",
          f"€{result['std_rate']:,.0f}",
          help="Language rate × vendor modifier. Applied before markup.")
p4.metric("LipSync Share",
          f"{result['pct']['LS']*100:.1f}%",
          help="Higher LipSync proportion increases minimum fee exposure.")

# ── NARRATIVE ─────────────────────────────────────────────────────────────────
gap = result['model_cost'] - result['classic_cost']
gap_pct = gap / result['classic_cost'] * 100 if result['classic_cost'] else 0

if abs(gap_pct) < 3:
    st.info(
        f"The classic and model estimates are closely aligned ({gap_pct:+.1f}%). "
        f"This session has a relatively standard TR mix and actor count, "
        f"where the flat-rate formula performs near its best."
    )
elif gap > 0:
    st.warning(
        f"The model estimates **€{gap:,.0f} more** than the classic forecast "
        f"({gap_pct:+.1f}%). "
        f"The classic formula is likely underestimating this session — "
        f"possibly due to a high proportion of restricted content "
        f"({result['pct']['LS']*100:.0f}% LipSync) combined with "
        f"{n_actors} actors, which increases minimum fee exposure."
    )
else:
    st.success(
        f"The model estimates **€{abs(gap):,.0f} less** than the classic forecast "
        f"({gap_pct:+.1f}%). "
        f"The 1.10 buffer in the classic formula is overcompensating here. "
        f"The session has a straightforward structure that the model prices more precisely."
    )
