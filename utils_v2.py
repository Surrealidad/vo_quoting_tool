"""
utils_v2.py — shared logic for the VO Quoting app (two-stage version).

All domain constants, model loaders, and prediction functions live here.
Pages import from this module — no business logic in page files.
"""

import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
MODEL_DIR  = ROOT / "models"
DATA_DIR   = ROOT / "data"

# ── DOMAIN CONSTANTS ──────────────────────────────────────────────────────────
TR_MULT = {'LS': 5.0, 'SS': 4.0, 'HTR': 2.5, 'STR': 2.0, 'NoTR': 1.0}
TR_KEYS = ['LS', 'SS', 'HTR', 'STR', 'NoTR']
TR_LABELS = {
    'LS':  'LipSync',
    'SS':  'SoundSync',
    'HTR': 'Hard Restriction',
    'STR': 'Soft Restriction',
    'NoTR': 'No Restriction (Wild)',
}

WORDS_PER_HR  = 1000
MARKUP        = 0.10
DEFAULT_FX    = 1.18   # latest rate in dataset

BASE_RATE_EUR = 400.0
LANGUAGE_RATES = {
    'Polish': 0.50, 'LATAM': 0.60, 'Brazilian Portuguese': 0.65,
    'Chinese': 0.70, 'Spanish': 0.85, 'German': 1.00, 'French': 1.05,
    'English': 1.10, 'Dutch': 1.15, 'Korean': 1.30, 'Japanese': 2.50,
}

VENDOR_INFO = {
    'GridMark':    {'region': 'EU',   'rate_mod': 1.00, 'min_fee_h': 1},
    'LarkFarm':    {'region': 'EU',   'rate_mod': 0.95, 'min_fee_h': 1},
    'SonicBridge': {'region': 'EU',   'rate_mod': 1.05, 'min_fee_h': 2},
    'EuroVox':     {'region': 'EU',   'rate_mod': 0.98, 'min_fee_h': 1},
    'MediaBooth':  {'region': 'EU',   'rate_mod': 1.02, 'min_fee_h': 2},
    'TokyoVox':    {'region': 'APAC', 'rate_mod': 1.00, 'min_fee_h': 1},
    'AsiaSound':   {'region': 'APAC', 'rate_mod': 0.92, 'min_fee_h': 1},
    'PacificAudio':{'region': 'APAC', 'rate_mod': 1.08, 'min_fee_h': 2},
    'LatinaVox':   {'region': 'LATAM','rate_mod': 1.00, 'min_fee_h': 1},
    'SoundHouse':  {'region': 'LATAM','rate_mod': 0.97, 'min_fee_h': 1},
}

REGION_LANGS = {
    'EU':   ['German', 'French', 'Spanish', 'English', 'Dutch', 'Polish'],
    'APAC': ['Japanese', 'Korean', 'Chinese'],
    'LATAM':['Brazilian Portuguese', 'LATAM'],
}
LANG_REGION = {
    lang: region
    for region, langs in REGION_LANGS.items()
    for lang in langs
}

VIP_RATE_MULTIPLIER = 2.5   # mid-point of 2×–3× range used in generation

# ── BILLING RULE ──────────────────────────────────────────────────────────────

def bill_hours(raw_hours: float, min_fee_h: int) -> int:
    """
    VO billing convention: 15-minute grace period.
    Fractional hours < 0.25 are dropped; >= 0.25 round up to next integer.
    Vendor minimum fee applied on top.
    """
    floored  = int(raw_hours)
    fraction = raw_hours - floored
    billed   = floored + 1 if fraction >= 0.25 else max(1, floored)
    return max(min_fee_h, billed)

# ── CACHED LOADERS ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_s1():
    return joblib.load(MODEL_DIR / "model_s1.pkl")

@st.cache_resource
def load_encoder_s1():
    return joblib.load(MODEL_DIR / "encoder_s1.pkl")

@st.cache_resource
def load_metadata_s1():
    with open(MODEL_DIR / "metadata_s1.json") as f:
        return json.load(f)

@st.cache_resource
def load_model_s2():
    return joblib.load(MODEL_DIR / "model_s2.pkl")

@st.cache_resource
def load_encoder_s2():
    return joblib.load(MODEL_DIR / "encoder_s2.pkl")

@st.cache_resource
def load_actor_stats():
    return joblib.load(MODEL_DIR / "actor_stats_s2.pkl")

@st.cache_resource
def load_metadata_s2():
    with open(MODEL_DIR / "metadata_s2.json") as f:
        return json.load(f)

@st.cache_data
def load_actor_reference():
    return pd.read_csv(DATA_DIR / "actor_reference.csv")

# ── VENDOR HELPERS ────────────────────────────────────────────────────────────

def vendors_for_language(language: str) -> list[str]:
    """Return vendors that operate in the given language's region."""
    region = LANG_REGION.get(language, 'EU')
    return [v for v, info in VENDOR_INFO.items() if info['region'] == region]

def actors_for_vendor_language(vendor: str, language: str) -> list[str]:
    """Return known actor names for a given vendor+language combination."""
    ref = load_actor_reference()
    subset = ref[(ref['Vendor'] == vendor) & (ref['Language'] == language)]
    return sorted(subset['ActorName'].tolist())

def actor_hist_stats(actor_name: str, vendor: str, language: str,
                     actor_stats: pd.DataFrame, global_mean: float,
                     global_std: float) -> tuple[float, int, float]:
    """
    Look up historical efficiency stats for a named actor.
    Returns (hist_ratio, session_count, hist_std).
    Falls back to global mean for unknown actors.
    """
    ref = load_actor_reference()
    match = ref[
        (ref['ActorName'] == actor_name) &
        (ref['Vendor']    == vendor)     &
        (ref['Language']  == language)
    ]
    if match.empty:
        return global_mean, 0, global_std

    actor_id = match.iloc[0]['ActorID']
    stats_row = actor_stats[actor_stats['ActorID'] == actor_id]
    if stats_row.empty:
        return global_mean, 0, global_std

    row = stats_row.iloc[0]
    return (
        float(row['Actor_Hist_Ratio']),
        int(row['Actor_Session_Count']),
        float(row.get('Actor_Hist_Std', global_std)),
    )

# ── STAGE 1 PREDICTION ────────────────────────────────────────────────────────

def predict_stage1(
    language: str, vendor: str, is_sequel: bool,
    tr_words: dict,   # {'LS': int, 'SS': int, ...}
    n_actors: int,
    exchange_rate: float,
) -> dict:
    """
    Stage 1 prediction — session level, no cast information.
    Returns dict with classic forecast, model prediction, and derived metrics.
    """
    model    = load_model_s1()
    encoder  = load_encoder_s1()
    metadata = load_metadata_s1()
    vi       = VENDOR_INFO[vendor]

    total_raw  = sum(tr_words.values())
    norm_wc    = int(sum(tr_words[tr] * TR_MULT[tr] for tr in TR_KEYS))
    est_hours  = bill_hours(norm_wc / WORDS_PER_HR, vi['min_fee_h']) * n_actors
    std_rate   = BASE_RATE_EUR * LANGUAGE_RATES[language] * vi['rate_mod']

    pct = {tr: tr_words[tr] / max(1, total_raw) for tr in TR_KEYS}

    # Classic S1: total hours × standard rate × 1.10 buffer
    classic_cost = round(est_hours * std_rate * 1.10, 2)
    classic_bill = round(classic_cost * (1 + MARKUP) * exchange_rate, 2)

    # Model prediction
    row = pd.DataFrame([{
        'Number_of_Actors':      n_actors,
        'Normalized_Wordcount':  norm_wc,
        'Estimated_Hours':       est_hours,
        'Standard_Rate_per_Hour': std_rate,
        'Pct_LS':  pct['LS'],
        'Pct_SS':  pct['SS'],
        'Pct_HTR': pct['HTR'],
        'Pct_STR': pct['STR'],
        'Is_Sequel': int(is_sequel),
        'Language': language,
        'Vendor':   vendor,
    }])
    cats = ['Language', 'Vendor']
    row[cats] = encoder.transform(row[cats])
    row = row[metadata['features']]

    model_cost = round(float(model.predict(row)[0]), 2)
    model_bill = round(model_cost * (1 + MARKUP) * exchange_rate, 2)

    return {
        'classic_cost': classic_cost,
        'classic_bill': classic_bill,
        'model_cost':   model_cost,
        'model_bill':   model_bill,
        'norm_wc':      norm_wc,
        'est_hours':    est_hours,
        'std_rate':     std_rate,
        'pct':          pct,
    }

# ── STAGE 2 PREDICTION ────────────────────────────────────────────────────────

def predict_stage2_actor(
    actor_name: str, vendor: str, language: str,
    is_vip: bool, is_sequel: bool,
    actor_tr_words: dict,   # {'LS': int, 'SS': int, ...}
    exchange_rate: float,
) -> dict:
    """
    Stage 2 prediction for one actor row.
    Returns dict with classic forecast, model prediction, and actor stats.
    """
    model       = load_model_s2()
    encoder     = load_encoder_s2()
    actor_stats = load_actor_stats()
    metadata    = load_metadata_s2()
    vi          = VENDOR_INFO[vendor]

    global_mean = metadata['global_mean_ratio']
    global_std  = metadata.get('global_hist_std', 0.20)

    norm_wc   = int(sum(actor_tr_words[tr] * TR_MULT[tr] for tr in TR_KEYS))
    est_hrs   = bill_hours(norm_wc / WORDS_PER_HR, vi['min_fee_h'])
    rate_mult = VIP_RATE_MULTIPLIER if is_vip else 1.0
    actor_rate = BASE_RATE_EUR * LANGUAGE_RATES[language] * vi['rate_mod'] * rate_mult

    # Classic S2: known cast, known rates, no efficiency knowledge
    classic_cost = round(est_hrs * actor_rate, 2)
    classic_bill = round(classic_cost * (1 + MARKUP) * exchange_rate, 2)

    # Actor historical stats
    hist_ratio, sess_count, hist_std = actor_hist_stats(
        actor_name, vendor, language, actor_stats, global_mean, global_std
    )

    # Get ActorID for encoding
    ref   = load_actor_reference()
    match = ref[
        (ref['ActorName'] == actor_name) &
        (ref['Vendor']    == vendor)     &
        (ref['Language']  == language)
    ]
    actor_id = match.iloc[0]['ActorID'] if not match.empty else 'UNKNOWN'

    row = pd.DataFrame([{
        'LS_Words':            actor_tr_words['LS'],
        'SS_Words':            actor_tr_words['SS'],
        'HTR_Words':           actor_tr_words['HTR'],
        'STR_Words':           actor_tr_words['STR'],
        'NoTR_Words':          actor_tr_words['NoTR'],
        'Normalized_Wordcount': norm_wc,
        'Estimated_Hours':     est_hrs,
        'Cost_per_Hour':       actor_rate,
        'Cost_Forecast_S2':    classic_cost,
        'Is_VIP':              int(is_vip),
        'Is_Sequel':           int(is_sequel),
        'Actor_Session_Count': sess_count,
        'Actor_Hist_Ratio':    hist_ratio,
        'Actor_Hist_Std':      hist_std,
        'Language':            language,
        'Vendor':              vendor,
    }])

    cats = ['Language', 'Vendor']
    row[cats] = encoder.transform(row[cats])
    row = row[metadata['features']]

    # Model predicts ratio; final cost = ratio × classic
    ratio_pred = float(model.predict(row)[0])
    model_cost = round(ratio_pred * classic_cost, 2)
    model_bill = round(model_cost * (1 + MARKUP) * exchange_rate, 2)

    return {
        'actor_name':    actor_name,
        'norm_wc':       norm_wc,
        'est_hours':     est_hrs,
        'actor_rate':    actor_rate,
        'classic_cost':  classic_cost,
        'classic_bill':  classic_bill,
        'model_cost':    model_cost,
        'model_bill':    model_bill,
        'hist_ratio':    hist_ratio,
        'sess_count':    sess_count,
        'is_known':      actor_id != 'UNKNOWN',
        'is_vip':        is_vip,
    }

def predict_stage2_session(actor_results: list[dict], exchange_rate: float) -> dict:
    """Aggregate actor-level Stage 2 predictions to session totals."""
    classic_cost = sum(r['classic_cost'] for r in actor_results)
    model_cost   = sum(r['model_cost']   for r in actor_results)
    return {
        'classic_cost': round(classic_cost, 2),
        'classic_bill': round(classic_cost * (1 + MARKUP) * exchange_rate, 2),
        'model_cost':   round(model_cost,   2),
        'model_bill':   round(model_cost   * (1 + MARKUP) * exchange_rate, 2),
        'n_actors':     len(actor_results),
        'n_known':      sum(r['is_known'] for r in actor_results),
        'n_vip':        sum(r['is_vip']   for r in actor_results),
    }

# ── CSV TEMPLATE ──────────────────────────────────────────────────────────────

def actor_template_df(n_rows: int = 5) -> pd.DataFrame:
    """Return an empty actor matrix template for download."""
    return pd.DataFrame({
        'ActorName': [''] * n_rows,
        'Is_VIP':    [False] * n_rows,
        'LS_Words':  [0] * n_rows,
        'SS_Words':  [0] * n_rows,
        'HTR_Words': [0] * n_rows,
        'STR_Words': [0] * n_rows,
        'NoTR_Words':[0] * n_rows,
    })
