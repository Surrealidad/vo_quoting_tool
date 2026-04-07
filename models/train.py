"""
VO Session Cost — Model Training Script
Run once to produce model.pkl, encoder.pkl, metadata.json, and diagnostic plots.
Usage: python models/train.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

# ─── PATHS ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "vo_dataset.csv"
MODEL_DIR = ROOT / "models"

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────

SPREAD_ORDER = ['Very Low', 'Low', 'Normal', 'High', 'Very High']
TR_ORDER     = ['NoTR', 'STR', 'HTR', 'SoundSync', 'LipSync']

CATEGORICAL  = ['Language', 'Time_Restriction']
NUMERIC      = [
    'Wordcount', 'Filecount',
    'Words_per_File', 'Words_per_Actor', 'Chars_per_Actor',
    'Number_of_Actors', 'Number_of_Characters',
    'Spread_Score', 'TR_Score',
]
FEATURES     = NUMERIC + CATEGORICAL
TARGET       = 'Cost_Actuals'

# ─── LOAD & FILTER ─────────────────────────────────────────────────────────────

print("Loading data...")
df_raw = pd.read_csv(DATA_PATH)
df = df_raw[df_raw[TARGET].notna()].copy()
print(f"  {len(df):,} past sessions available for training (of {len(df_raw):,} total)")

# ─── FEATURE ENGINEERING ───────────────────────────────────────────────────────

df['Words_per_File']  = df['Wordcount']          / df['Filecount']
df['Words_per_Actor'] = df['Wordcount']           / df['Number_of_Actors']
df['Chars_per_Actor'] = df['Number_of_Characters']/ df['Number_of_Actors']
df['Spread_Score']    = df['Scope_Spread'].map({s: i for i, s in enumerate(SPREAD_ORDER)})
df['TR_Score']        = df['Time_Restriction'].map({t: i for i, t in enumerate(TR_ORDER)})

# ─── ENCODE CATEGORICALS ───────────────────────────────────────────────────────

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[CATEGORICAL] = encoder.fit_transform(df[CATEGORICAL])

# ─── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─── CLASSIC BASELINE (on test set) ────────────────────────────────────────
# Word-based classic: wordcount × base_rate × TR_mult × language_rate.
# A competent per-word approach — better than file-based, but blind to
# spread, minimum fees, and words-per-actor dynamics.

BASE_WORD_RATE   = 1.30
TR_MULT_CLASSIC  = {'NoTR': 1.0, 'STR': 2.0, 'HTR': 2.5, 'SoundSync': 3.5, 'LipSync': 4.0}
LANGUAGE_RATES   = {
    'Polish': 0.50, 'LATAM': 0.60, 'Brazilian Portuguese': 0.65,
    'Chinese': 0.70, 'Spanish': 0.85, 'German': 1.00, 'French': 1.05,
    'English': 1.10, 'Dutch': 1.15, 'Korean': 1.30, 'Japanese': 2.50,
}

# Compute classic forecast on the full past dataset BEFORE encoding,
# then index into the test set after the split.
df_raw_past = df_raw[df_raw[TARGET].notna()].copy()
df_raw_past['classic'] = df_raw_past.apply(
    lambda r: r['Wordcount']
              * BASE_WORD_RATE
              * TR_MULT_CLASSIC[r['Time_Restriction']]
              * LANGUAGE_RATES[r['Language']],
    axis=1
)

test_idx      = X_test.index
y_classic     = df_raw_past.loc[test_idx, 'classic']
mape_baseline = mean_absolute_percentage_error(y_test, y_classic) * 100
print(f"\nClassic baseline MAPE (word-based): {mape_baseline:.1f}%")

# ─── MODEL TRAINING ────────────────────────────────────────────────────────────

print("\nTraining XGBoost...")
model = xgb.XGBRegressor(
    n_estimators     = 500,
    learning_rate    = 0.04,
    max_depth        = 5,
    subsample        = 0.80,
    colsample_bytree = 0.80,
    min_child_weight = 5,
    reg_alpha        = 0.05,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

model.fit(
    X_train, y_train,
    eval_set        = [(X_test, y_test)],
    verbose         = False,
)

# ─── EVALUATION ────────────────────────────────────────────────────────────────

y_pred     = model.predict(X_test)
mape_model = mean_absolute_percentage_error(y_test, y_pred) * 100
r2_model   = r2_score(y_test, y_pred)

# Cross-validated MAPE (5-fold on full training set)
cv_scores  = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='neg_mean_absolute_percentage_error')
mape_cv    = -cv_scores.mean() * 100

# MAPE by TR type
df_test              = df.loc[test_idx].copy()
df_test['y_pred']    = y_pred
df_test['y_true']    = y_test.values
df_test['y_classic'] = y_classic.values
df_test['mape_row']  = (abs(df_test['y_pred']    - df_test['y_true']) / df_test['y_true']) * 100
df_test['mape_base'] = (abs(df_test['y_classic'] - df_test['y_true']) / df_test['y_true']) * 100
mape_by_tr           = df_test.groupby('Time_Restriction')[['mape_row','mape_base']].mean()

print(f"\n{'─'*50}")
print(f"  Classic MAPE (test)  : {mape_baseline:.1f}%")
print(f"  XGBoost MAPE (test)      : {mape_model:.1f}%")
print(f"  XGBoost MAPE (5-fold CV) : {mape_cv:.1f}%")
print(f"  R²                       : {r2_model:.3f}")
print(f"  Improvement              : {mape_baseline - mape_model:.1f} pp")
print(f"{'─'*50}")
print(f"\nMAPE by TR type (test set):")
print(f"  {'TR Type':<12} {'Classic':>12} {'XGBoost':>10}")
print(f"  {'─'*36}")
tr_labels = {i: t for i, t in enumerate(TR_ORDER)}
for tr_enc, row in mape_by_tr.iterrows():
    tr_name = tr_labels.get(int(tr_enc), str(tr_enc))
    print(f"  {tr_name:<12} {row['mape_base']:>11.1f}%  {row['mape_row']:>8.1f}%")

# ─── PLOTS ─────────────────────────────────────────────────────────────────────

print("\nGenerating diagnostic plots...")

# 1 — SHAP summary (beeswarm)
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, feature_names=FEATURES, show=False, plot_size=None)
plt.title("SHAP Feature Importance — VO Cost Model", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(MODEL_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()

# 2 — Predicted vs Actual scatter
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (y_hat, label, color) in zip(axes, [
    (y_classic.values, "Classic Forecast", "#d62728"),
    (y_pred,               "XGBoost Prediction",   "#1f77b4"),
]):
    ax.scatter(y_test, y_hat, alpha=0.35, s=18, color=color)
    lim = max(y_test.max(), np.array(y_hat).max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label="Perfect prediction")
    ax.set_xlabel("Actual Cost (EUR)", fontsize=10)
    ax.set_ylabel("Predicted Cost (EUR)", fontsize=10)
    mape_val = mean_absolute_percentage_error(y_test, y_hat) * 100
    ax.set_title(f"{label}\nMAPE = {mape_val:.1f}%", fontsize=11)
    ax.legend(fontsize=8)
    ax.ticklabel_format(style='plain', axis='both')

plt.suptitle("Predicted vs Actual — VO Session Cost (EUR)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(MODEL_DIR / "pred_vs_actual.png", dpi=150, bbox_inches='tight')
plt.close()

# 3 — MAPE comparison by TR type (bar chart)
tr_names  = [tr_labels.get(int(i), str(i)) for i in mape_by_tr.index]
x         = np.arange(len(tr_names))
width     = 0.35

fig, ax   = plt.subplots(figsize=(8, 4))
ax.bar(x - width/2, mape_by_tr['mape_base'], width, label='Classic', color='#d62728', alpha=0.8)
ax.bar(x + width/2, mape_by_tr['mape_row'],  width, label='XGBoost',     color='#1f77b4', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(tr_names)
ax.set_ylabel("MAPE (%)")
ax.set_title("MAPE by Time Restriction — Classic vs XGBoost")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(MODEL_DIR / "mape_by_tr.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Plots saved to {MODEL_DIR}")

# ─── SAVE ARTEFACTS ────────────────────────────────────────────────────────────

joblib.dump(model,   MODEL_DIR / "model.pkl")
joblib.dump(encoder, MODEL_DIR / "encoder.pkl")

metadata = {
    "features":            FEATURES,
    "numeric_features":    NUMERIC,
    "categorical_features":CATEGORICAL,
    "spread_order":        SPREAD_ORDER,
    "tr_order":            TR_ORDER,
    "encoder_categories":  {
        col: list(cats)
        for col, cats in zip(CATEGORICAL, encoder.categories_)
    },
    "metrics": {
        "mape_classic": round(mape_baseline, 2),
        "mape_model_test":  round(mape_model, 2),
        "mape_model_cv":    round(mape_cv, 2),
        "r2":               round(r2_model, 3),
        "improvement_pp":   round(mape_baseline - mape_model, 2),
    },
    "training_samples": int(len(X_train)),
    "test_samples":     int(len(X_test)),
}

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ model.pkl, encoder.pkl, metadata.json saved to {MODEL_DIR}")
print("  Training complete.")
