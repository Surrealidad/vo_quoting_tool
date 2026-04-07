"""
pages/02_stage2.py — Stage 2 Actor-Level Predictor
Includes Stage 1 session estimate at top, then actor matrix for Stage 2.
"""

import io
import streamlit as st
import pandas as pd
from utils_v2 import (
    predict_stage1, predict_stage2_actor, predict_stage2_session,
    vendors_for_language, actors_for_vendor_language,
    actor_template_df, load_actor_stats, load_metadata_s2,
    LANGUAGE_RATES, TR_LABELS, TR_KEYS, DEFAULT_FX,
)

st.title("🎙️ Stage 2 — Actor-Level Estimate")
st.caption(
    "Use this when the cast is confirmed and per-actor scope is available. "
    "Builds on the Stage 1 session estimate by incorporating actor history, "
    "individual rates, and VIP fees."
)
st.divider()

# ── SESSION INPUTS (shared with Stage 1) ──────────────────────────────────────
st.subheader("Session parameters")

col1, col2, col3 = st.columns(3)
with col1:
    language = st.selectbox(
        "Language",
        options=sorted(LANGUAGE_RATES.keys()),
        index=sorted(LANGUAGE_RATES.keys()).index("German"),
        key="s2_lang",
    )
with col2:
    vendor_opts = vendors_for_language(language)
    vendor = st.selectbox("Vendor", options=vendor_opts, key="s2_vendor")
with col3:
    exchange_rate = st.number_input(
        "Exchange Rate (EUR → USD)", min_value=0.50, max_value=3.00,
        value=DEFAULT_FX, step=0.01, format="%.2f", key="s2_fx",
    )

col4, col5 = st.columns(2)
with col4:
    n_actors_est = st.number_input(
        "Estimated Number of Actors", min_value=1, max_value=200,
        value=10, step=1, key="s2_n_actors",
        help="Used for Stage 1 estimate. Stage 2 uses the actual actor matrix.",
    )
with col5:
    is_sequel = st.checkbox("This is a sequel project", value=False, key="s2_sequel")

st.divider()
st.subheader("Script scope by time restriction")
st.caption("Total words per restriction type across the full session.")

tr_cols = st.columns(5)
tr_words = {}
for i, tr in enumerate(TR_KEYS):
    with tr_cols[i]:
        tr_words[tr] = st.number_input(
            TR_LABELS[tr], min_value=0, max_value=500_000,
            value=10000 if tr == 'NoTR' else 0, step=500,
            key=f"s2_tr_{tr}",
        )

total_words = sum(tr_words.values())

# ── STAGE 1 RESULT (shown immediately) ────────────────────────────────────────
if total_words > 0:
    s1 = predict_stage1(
        language=language, vendor=vendor, is_sequel=is_sequel,
        tr_words=tr_words, n_actors=n_actors_est, exchange_rate=exchange_rate,
    )
    with st.expander("📋 Stage 1 estimate (session level)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Classic (EUR)", f"€{s1['classic_cost']:,.0f}")
        c2.metric("Model (EUR)",   f"€{s1['model_cost']:,.0f}",
                  delta=f"€{s1['model_cost']-s1['classic_cost']:+,.0f}",
                  delta_color="off")
        c3.metric("Classic Billing (USD)", f"${s1['classic_bill']:,.0f}")
        c4.metric("Model Billing (USD)",   f"${s1['model_bill']:,.0f}")
        st.caption(
            f"Normalised wordcount: {s1['norm_wc']:,} | "
            f"Estimated hours: {s1['est_hours']:,} | "
            f"Standard rate: €{s1['std_rate']:,.0f}/hr"
        )
else:
    st.info("Enter word counts above to generate estimates.")
    st.stop()

st.divider()

# ── ACTOR MATRIX ──────────────────────────────────────────────────────────────
st.subheader("Actor scope matrix")
st.caption(
    "Enter one row per actor. Words are the actor's individual scope "
    "within this session, split by time restriction type. "
    "Actor names autocomplete from the known pool for this vendor and language."
)

# CSV template download
template_csv = actor_template_df(5).to_csv(index=False).encode()
st.download_button(
    "⬇️ Download CSV template",
    data=template_csv,
    file_name="actor_scope_template.csv",
    mime="text/csv",
    help="Fill in Excel or Sheets, then upload below.",
)

# CSV upload
uploaded = st.file_uploader(
    "Upload completed CSV (optional — or fill the matrix directly below)",
    type="csv", key="s2_upload",
)

# Known actors for autocomplete
known_actors = actors_for_vendor_language(vendor, language)
actor_col_config = st.column_config.SelectboxColumn(
    "Actor Name",
    options=known_actors,
    required=False,
    help="Select from known actors for history lookup, or type a new name.",
) if known_actors else st.column_config.TextColumn("Actor Name")

# Build initial dataframe
if uploaded is not None:
    try:
        init_df = pd.read_csv(uploaded)
        required = ['ActorName','Is_VIP','LS_Words','SS_Words',
                    'HTR_Words','STR_Words','NoTR_Words']
        missing = [c for c in required if c not in init_df.columns]
        if missing:
            st.error(f"CSV missing columns: {missing}. Download the template above.")
            init_df = actor_template_df(5)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        init_df = actor_template_df(5)
else:
    init_df = actor_template_df(5)

# Editable matrix
edited_df = st.data_editor(
    init_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "ActorName": actor_col_config,
        "Is_VIP":    st.column_config.CheckboxColumn("VIP", default=False),
        "LS_Words":  st.column_config.NumberColumn("LipSync Words",     min_value=0, default=0),
        "SS_Words":  st.column_config.NumberColumn("SoundSync Words",   min_value=0, default=0),
        "HTR_Words": st.column_config.NumberColumn("HTR Words",         min_value=0, default=0),
        "STR_Words": st.column_config.NumberColumn("STR Words",         min_value=0, default=0),
        "NoTR_Words":st.column_config.NumberColumn("NoTR (Wild) Words", min_value=0, default=0),
    },
    key="actor_matrix",
)

# ── STAGE 2 PREDICTION ────────────────────────────────────────────────────────
valid_rows = edited_df[
    edited_df['ActorName'].notna() &
    (edited_df['ActorName'].astype(str).str.strip() != '') &
    (edited_df[['LS_Words','SS_Words','HTR_Words','STR_Words','NoTR_Words']].sum(axis=1) > 0)
].copy()

if valid_rows.empty:
    st.info("Fill in at least one actor row with a name and word counts to generate the Stage 2 estimate.")
    st.stop()

st.divider()
st.subheader("Stage 2 results")

# Run predictions per actor
actor_results = []
for _, row in valid_rows.iterrows():
    actor_tr = {
        'LS':   int(row.get('LS_Words',  0) or 0),
        'SS':   int(row.get('SS_Words',  0) or 0),
        'HTR':  int(row.get('HTR_Words', 0) or 0),
        'STR':  int(row.get('STR_Words', 0) or 0),
        'NoTR': int(row.get('NoTR_Words',0) or 0),
    }
    res = predict_stage2_actor(
        actor_name=str(row['ActorName']).strip(),
        vendor=vendor, language=language,
        is_vip=bool(row.get('Is_VIP', False)),
        is_sequel=is_sequel,
        actor_tr_words=actor_tr,
        exchange_rate=exchange_rate,
    )
    actor_results.append(res)

session_totals = predict_stage2_session(actor_results, exchange_rate)

# Session totals
st.markdown("**Session totals**")
t1, t2, t3, t4 = st.columns(4)
t1.metric("S2 Classic (EUR)", f"€{session_totals['classic_cost']:,.0f}")
t2.metric("S2 Model (EUR)",   f"€{session_totals['model_cost']:,.0f}",
          delta=f"€{session_totals['model_cost']-session_totals['classic_cost']:+,.0f}",
          delta_color="off")
t3.metric("S2 Classic Billing (USD)", f"${session_totals['classic_bill']:,.0f}")
t4.metric("S2 Model Billing (USD)",   f"${session_totals['model_bill']:,.0f}")

st.caption(
    f"{session_totals['n_actors']} actors | "
    f"{session_totals['n_known']} with known history | "
    f"{session_totals['n_vip']} VIP"
)

# Stage 1 vs Stage 2 comparison
with st.expander("Stage 1 vs Stage 2 comparison", expanded=False):
    comp_cols = st.columns(2)
    with comp_cols[0]:
        st.markdown("**Classic method**")
        st.markdown(f"Stage 1: €{s1['classic_cost']:,.0f} / ${s1['classic_bill']:,.0f}")
        st.markdown(f"Stage 2: €{session_totals['classic_cost']:,.0f} / ${session_totals['classic_bill']:,.0f}")
        diff_c = session_totals['classic_cost'] - s1['classic_cost']
        st.caption(f"Difference: €{diff_c:+,.0f} — driven by VIP rates not captured in Stage 1")
    with comp_cols[1]:
        st.markdown("**ML model**")
        st.markdown(f"Stage 1: €{s1['model_cost']:,.0f} / ${s1['model_bill']:,.0f}")
        st.markdown(f"Stage 2: €{session_totals['model_cost']:,.0f} / ${session_totals['model_bill']:,.0f}")
        diff_m = session_totals['model_cost'] - s1['model_cost']
        st.caption(f"Difference: €{diff_m:+,.0f} — actor history and VIP rates refined at actor level")

# Per-actor breakdown table
st.markdown("**Per-actor breakdown**")
actor_table = pd.DataFrame([{
    'Actor':          r['actor_name'],
    'VIP':            '✓' if r['is_vip'] else '',
    'Known':          '✓' if r['is_known'] else '—',
    'Hist. Ratio':    f"{r['hist_ratio']:.2f}x",
    'Est. Hours':     r['est_hours'],
    'Rate (€/hr)':    f"€{r['actor_rate']:,.0f}",
    'Classic (€)':    f"€{r['classic_cost']:,.0f}",
    'Model (€)':      f"€{r['model_cost']:,.0f}",
    'Δ (€)':          f"€{r['model_cost']-r['classic_cost']:+,.0f}",
} for r in actor_results])

st.dataframe(actor_table, use_container_width=True, hide_index=True)

# Actor history note
n_unknown = session_totals['n_actors'] - session_totals['n_known']
if n_unknown > 0:
    st.caption(
        f"{n_unknown} actor(s) marked '—' under Known have no session history "
        f"in the training data. Their efficiency is estimated at the global average "
        f"({load_metadata_s2()['global_mean_ratio']:.2f}x)."
    )
