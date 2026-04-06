"""
utils.py — shared helpers for the VO Quoting app.

All feature engineering lives here so that the app and train.py
stay consistent. If you change a feature definition, change it here
and update train.py to match (or vice versa).
"""

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

# ─── PATHS ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "vo_dataset.csv"

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────

SPREAD_ORDER = ['Very Low', 'Low', 'Normal', 'High', 'Very High']
TR_ORDER     = ['NoTR', 'STR', 'HTR', 'SoundSync', 'LipSync']

# Classic rates: EUR per file, per TR type.
# Calibrated at a plausible but slightly-off words-per-file assumption —
# this is what the "old method" does. See train.py for full explanation.
CLASSIC_RATES = {
    'NoTR':      1.30 * 1.0 * 7.0,
    'STR':       1.30 * 2.0 * 6.5,
    'HTR':       1.30 * 2.5 * 6.0,
    'SoundSync': 1.30 * 3.5 * 5.5,
    'LipSync':   1.30 * 4.0 * 5.5,
}

LANGUAGE_RATES = {
    'Polish': 0.50, 'LATAM': 0.60, 'Brazilian Portuguese': 0.65,
    'Chinese': 0.70, 'Spanish': 0.85, 'German': 1.00, 'French': 1.05,
    'English': 1.10, 'Dutch': 1.15, 'Korean': 1.30, 'Japanese': 2.50,
}

# ─── CACHED LOADERS ────────────────────────────────────────────────────────────
# @st.cache_resource means these are loaded once per app session,
# not on every user interaction. Critical for performance.

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load(MODEL_DIR / "encoder.pkl")

@st.cache_resource
def load_metadata():
    with open(MODEL_DIR / "metadata.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ─── FEATURE ENGINEERING ───────────────────────────────────────────────────────
# Must mirror train.py exactly. The model was trained on these derived features —
# any mismatch between training and inference will silently corrupt predictions.

def build_features(wordcount, filecount, n_actors, n_chars,
                   language, time_restriction, scope_spread):
    """
    Takes raw user inputs and returns a one-row DataFrame
    ready for encoder.transform() and model.predict().
    """
    wpf   = wordcount / filecount      # words per file  — key driver
    wpa   = wordcount / n_actors       # words per actor — proxy for scope depth
    cpa   = n_chars   / n_actors       # chars per actor — proxy for cast spread

    spread_score = SPREAD_ORDER.index(scope_spread)   # ordinal: 0–4
    tr_score     = TR_ORDER.index(time_restriction)   # ordinal: 0–4

    row = {
        # Raw inputs the model uses directly
        'Wordcount':            wordcount,
        'Filecount':            filecount,
        'Number_of_Actors':     n_actors,
        'Number_of_Characters': n_chars,
        # Engineered features
        'Words_per_File':       wpf,
        'Words_per_Actor':      wpa,
        'Chars_per_Actor':      cpa,
        'Spread_Score':         spread_score,
        'TR_Score':             tr_score,
        # Categoricals (will be encoded below)
        'Language':             language,
        'Time_Restriction':     time_restriction,
    }
    return pd.DataFrame([row])

def encode_and_predict(df_row, model, encoder, feature_order):
    """
    Applies the saved OrdinalEncoder to categorical columns,
    reorders columns to match training order, runs model.predict().

    Returns a single float (EUR cost estimate).
    """
    categoricals = ['Language', 'Time_Restriction']
    df_row[categoricals] = encoder.transform(df_row[categoricals])
    # Reorder to exactly match training feature list
    df_row = df_row[feature_order]
    return float(model.predict(df_row)[0])

# ─── CLASSIC FORMULA ───────────────────────────────────────────────────────

def classic_forecast(filecount, language, time_restriction):
    """
    The 'old method': filecount × rate per file.
    TR-aware and language-aware, but blind to wordcount,
    words-per-file variance, actor spread, and minimum fees.
    """
    rate = CLASSIC_RATES[time_restriction] * LANGUAGE_RATES[language]
    return round(filecount * rate, 2)
