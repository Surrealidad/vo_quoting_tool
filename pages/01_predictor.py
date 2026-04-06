"""
pages/01_predictor.py — Quote Predictor

User fills in script parameters → model returns cost estimate.
The classic forecast is computed on the same inputs and shown
below for direct comparison, with the gap clearly labelled.
"""

import streamlit as st
import numpy as np
from utils import (
    load_model, load_encoder, load_metadata,
    build_features, encode_and_predict, classic_forecast,
    SPREAD_ORDER, TR_ORDER, LANGUAGE_RATES,
)

# ─── LOAD ARTEFACTS ────────────────────────────────────────────────────────────

model    = load_model()
encoder  = load_encoder()
metadata = load_metadata()

# ─── PAGE HEADER ───────────────────────────────────────────────────────────────

st.title("🎙️ VO Session Cost Estimator")
st.caption(
    "Enter your script parameters below. The model will estimate the session "
    "cost in EUR based on patterns learned from historical data — going beyond "
    "the flat file-rate approach used in traditional quoting."
)
st.divider()

# ─── INPUT FORM ────────────────────────────────────────────────────────────────
# Two columns: left for scope/script params, right for session/cast params.
# Using st.form so the model only runs when the user explicitly clicks Estimate,
# not on every single widget interaction.

with st.form("quote_form"):
    st.subheader("Script Parameters")
    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Language",
            options=sorted(LANGUAGE_RATES.keys()),
            index=sorted(LANGUAGE_RATES.keys()).index("German"),
            help="Target recording language. Language rate is a significant cost driver."
        )
        time_restriction = st.selectbox(
            "Time Restriction",
            options=TR_ORDER,
            help=(
                "NoTR = no restriction (wild), STR = soft, HTR = hard, "
                "SoundSync = sound-synced, LipSync = lip-synced. "
                "LipSync is ~4× more expensive per word than NoTR."
            )
        )
        scope_spread = st.selectbox(
            "Scope Spread among Actors",
            options=SPREAD_ORDER,
            index=SPREAD_ORDER.index("Normal"),
            help=(
                "How evenly distributed is the scope across actors? "
                "Very High spread means many actors with thin scope — "
                "minimum fees kick in and drive cost up."
            )
        )

    with col2:
        wordcount = st.number_input(
            "Wordcount",
            min_value=100,
            max_value=150_000,
            value=10_000,
            step=500,
            help="Total word count in the script. The primary cost driver."
        )
        filecount = st.number_input(
            "Filecount",
            min_value=50,
            max_value=80_000,
            value=2_000,
            step=100,
            help=(
                "Number of files to deliver. Combined with wordcount, "
                "this gives words-per-file — a key feature the old method ignores."
            )
        )
        n_actors = st.number_input(
            "Number of Actors",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            help="Unique human actors attending the session."
        )
        n_chars = st.number_input(
            "Number of Characters",
            min_value=1,
            max_value=500,
            value=20,
            step=1,
            help="Game characters covered. Usually higher than actor count (one actor, multiple characters)."
        )

    submitted = st.form_submit_button("Estimate Cost", use_container_width=True, type="primary")

# ─── DERIVED METRICS PREVIEW ───────────────────────────────────────────────────
# Show the user the computed ratios before the prediction — this makes the
# feature engineering transparent and helps them understand the model's inputs.

if submitted:
    wpf = wordcount / filecount
    wpa = wordcount / n_actors

    st.divider()
    st.subheader("Session Profile")
    st.caption("Derived metrics computed from your inputs — these are what the model actually sees.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Words per File",  f"{wpf:.1f}",
              help="Average script density per file. LipSync sessions typically run 4–5 wpf, NoTR around 7–9.")
    m2.metric("Words per Actor", f"{wpa:,.0f}",
              help="Average scope depth per actor. Low values trigger minimum fee premiums.")
    m3.metric("Chars per Actor", f"{n_chars/n_actors:.1f}",
              help="How many characters each actor covers on average.")

    st.divider()

    # ─── PREDICTION ────────────────────────────────────────────────────────────

    df_row   = build_features(wordcount, filecount, n_actors, n_chars,
                               language, time_restriction, scope_spread)
    ml_cost  = encode_and_predict(df_row, model, encoder, metadata["features"])
    old_cost = classic_forecast(filecount, language, time_restriction)
    gap      = ml_cost - old_cost
    gap_pct  = (gap / old_cost) * 100

    # ─── RESULTS ───────────────────────────────────────────────────────────────

    st.subheader("Cost Estimate")

    r1, r2, r3 = st.columns(3)

    r1.metric(
        label="ML Model Estimate",
        value=f"€{ml_cost:,.0f}",
        help="XGBoost prediction based on wordcount, WPF, spread, language, and TR type."
    )
    r2.metric(
        label="Traditional Forecast",
        value=f"€{old_cost:,.0f}",
        help="Filecount × fixed rate per TR and language. Does not account for wordcount density or spread."
    )
    r3.metric(
        label="Difference",
        value=f"€{abs(gap):,.0f}",
        delta=f"{gap_pct:+.1f}% vs traditional",
        delta_color="inverse",
        help=(
            "Positive = traditional method underestimates. "
            "Negative = traditional method overestimates. "
            f"Overall, the traditional method is off by ~{metadata['metrics']['mape_classic']:.0f}% on average."
        )
    )

    # ─── CONTEXTUAL NARRATIVE ──────────────────────────────────────────────────
    # Explain the gap in plain English based on the session parameters.

    st.divider()

    if abs(gap_pct) < 5:
        msg = (
            f"The traditional and ML estimates are closely aligned for this session "
            f"({gap_pct:+.1f}%). This tends to happen when words-per-file is close to "
            f"the rate assumption the traditional method was calibrated on (~7 wpf for NoTR)."
        )
        st.info(msg)
    elif gap > 0:
        msg = (
            f"The traditional method **underestimates** this session by €{gap:,.0f} "
            f"({gap_pct:.1f}%). "
        )
        if wpf < 6:
            msg += (
                f"The main reason: this session has only **{wpf:.1f} words per file**, "
                f"but the traditional rate was calibrated assuming ~7 wpf. "
                f"Fewer words per file means more files to produce the same content — "
                f"and the traditional method mistakes filecount for workload."
            )
        if scope_spread in ("High", "Very High"):
            msg += (
                f" Scope spread is **{scope_spread}**, meaning several actors "
                f"have thin individual scopes — minimum fees are likely driving cost up."
            )
        st.warning(msg)
    else:
        msg = (
            f"The traditional method **overestimates** this session by €{abs(gap):,.0f} "
            f"({abs(gap_pct):.1f}%). "
            f"This session has **{wpf:.1f} words per file** — denser than the "
            f"calibration assumption. The model correctly recognises the higher "
            f"productivity per file and adjusts downward."
        )
        st.success(msg)
