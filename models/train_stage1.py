"""
Stage 1 Model — Session-Level Cost Prediction
Run from project root: python models/train_stage1.py

Context: At Stage 1 you know the session scope, TR mix, languages, and vendor
but NOT the cast. Classic baseline uses total estimated hours × standard rate × 1.10.
The model improves on this by learning patterns from TR mix, actor count, and
session-level features that the flat-rate formula ignores.
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

ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data' / 'vo_dataset_stage1.csv'
MODEL_DIR = ROOT / 'models'
TARGET    = 'Cost_Actuals_S1'

# ── FEATURES ──────────────────────────────────────────────────────────────────
CATEGORICALS = ['Language', 'Vendor']
NUMERICS     = [
    'LS_Words_Total', 'SS_Words_Total', 'HTR_Words_Total',
    'STR_Words_Total', 'NoTR_Words_Total',
    'Number_of_Actors',
    'Estimated_Hours',
    'Standard_Rate_per_Hour',
    'Is_Sequel',
]
FEATURES = NUMERICS + CATEGORICALS

# ── LOAD & FILTER ─────────────────────────────────────────────────────────────
print("Loading Stage 1 data...")
df = pd.read_csv(DATA_PATH)
df = df[df[TARGET].notna()].copy()
print(f"  {len(df):,} past sessions available")

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
df['Is_Sequel'] = df['Is_Sequel'].astype(int)

# ── ENCODE ────────────────────────────────────────────────────────────────────
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[CATEGORICALS] = encoder.fit_transform(df[CATEGORICALS])

# ── SPLIT ─────────────────────────────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── CLASSIC BASELINE ──────────────────────────────────────────────────────────
test_idx    = X_test.index
y_classic   = df.loc[test_idx, 'Cost_Forecast_S1']
mape_classic = mean_absolute_percentage_error(y_test, y_classic) * 100
print(f"\nClassic baseline MAPE: {mape_classic:.1f}%")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("\nTraining Stage 1 XGBoost...")
model = xgb.XGBRegressor(
    n_estimators     = 600,
    learning_rate    = 0.04,
    max_depth        = 5,
    subsample        = 0.80,
    colsample_bytree = 0.80,
    min_child_weight = 4,
    reg_alpha        = 0.05,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
y_pred     = model.predict(X_test)
mape_model = mean_absolute_percentage_error(y_test, y_pred) * 100
r2_model   = r2_score(y_test, y_pred)
cv_scores  = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='neg_mean_absolute_percentage_error')
mape_cv    = -cv_scores.mean() * 100

# Per-archetype breakdown
df_test              = df.loc[test_idx].copy()
df_test['y_pred']    = y_pred
df_test['y_true']    = y_test.values
df_test['y_classic'] = y_classic.values
df_test['mape_model']   = abs(df_test['y_pred']    - df_test['y_true']) / df_test['y_true'] * 100
df_test['mape_classic'] = abs(df_test['y_classic'] - df_test['y_true']) / df_test['y_true'] * 100

print(f"\n{'─'*52}")
print(f"  Classic MAPE (test)      : {mape_classic:.1f}%")
print(f"  Model MAPE  (test)       : {mape_model:.1f}%")
print(f"  Model MAPE  (5-fold CV)  : {mape_cv:.1f}%")
print(f"  R²                       : {r2_model:.3f}")
print(f"  Improvement              : {mape_classic - mape_model:.1f} pp")
print(f"{'─'*52}")
print(f"\nMAPE by archetype:")
arch_mape = df_test.groupby('Archetype')[['mape_classic','mape_model']].mean()
print(f"  {'Archetype':<14} {'Classic':>8} {'Model':>8}")
print(f"  {'─'*32}")
for arch, row in arch_mape.iterrows():
    print(f"  {arch:<14} {row['mape_classic']:>7.1f}%  {row['mape_model']:>7.1f}%")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# Archetype order for consistent display
ARCH_ORDER = ['mobile_tiny','mobile_small','indie','mid','aaa','mega']
ARCH_LABELS = {
    'mobile_tiny':  'Mobile\nTiny',
    'mobile_small': 'Mobile\nSmall',
    'indie':        'Indie',
    'mid':          'Mid',
    'aaa':          'AAA',
    'mega':         'Mega',
}

# ── Chart 1: MAPE by archetype ─────────────────────────────────────────────
arch_mape = df_test.groupby('Archetype')[['mape_classic','mape_model']].mean()
arch_mape = arch_mape.reindex([a for a in ARCH_ORDER if a in arch_mape.index])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
x = np.arange(len(arch_mape))
w = 0.35
bars_c = ax.bar(x - w/2, arch_mape['mape_classic'], w,
                label='Classic', color='#d62728', alpha=0.82)
bars_m = ax.bar(x + w/2, arch_mape['mape_model'],   w,
                label='Model',   color='#1f77b4', alpha=0.82)
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABELS.get(a, a) for a in arch_mape.index], fontsize=9)
ax.set_ylabel('MAPE (%)')
ax.set_title('Quoting Error by Project Size — Stage 1', fontsize=11)
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bar in list(bars_c) + list(bars_m):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=7.5)

# Right: EUR absolute error per session — two decimal places on labels
df_test['abs_err_classic'] = abs(df_test['y_classic'] - df_test['y_true'])
df_test['abs_err_model']   = abs(df_test['y_pred']    - df_test['y_true'])
arch_eur = df_test.groupby('Archetype')[['abs_err_classic','abs_err_model']].mean()
arch_eur = arch_eur.reindex([a for a in ARCH_ORDER if a in arch_eur.index])

ax2 = axes[1]
x2  = np.arange(len(arch_eur))
bars_c2 = ax2.bar(x2 - w/2, arch_eur['abs_err_classic']/1000, w,
                  label='Classic', color='#d62728', alpha=0.82)
bars_m2 = ax2.bar(x2 + w/2, arch_eur['abs_err_model']/1000,   w,
                  label='Model',   color='#1f77b4', alpha=0.82)
ax2.set_xticks(x2)
ax2.set_xticklabels([ARCH_LABELS.get(a, a) for a in arch_eur.index], fontsize=9)
ax2.set_ylabel('Mean Absolute Error (€k per session)')
ax2.set_title('EUR Quoting Error by Project Size — Stage 1', fontsize=11)
ax2.legend(); ax2.grid(axis='y', alpha=0.3)
for bar in list(bars_c2) + list(bars_m2):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f'€{bar.get_height():.2f}k', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'arch_breakdown_s1.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Chart 2: Budget accuracy by project (billing, aggregated to project) ──
# Budget accuracy = 1 - |total_billing_forecast - total_billing_actual|
#                       / total_billing_actual   (× 100, higher is better)
# Using billing (USD, with markup) to reflect what finance teams track.
# Aggregated to project level so session-level noise partially cancels out.

MARKUP_RATE = 0.10
df_test['billing_model'] = df_test['y_pred'] * (1 + MARKUP_RATE) * df_test['Exchange_Rate']

proj_agg = df_test.groupby(['ProjectID','Archetype']).agg(
    bill_classic=('Billing_Forecast_S1', 'sum'),
    bill_actual =('Billing_Actuals_S1',  'sum'),
    bill_model  =('billing_model',        'sum'),
).reset_index()

proj_agg['acc_classic'] = (
    1 - abs(proj_agg['bill_classic'] - proj_agg['bill_actual'])
        / proj_agg['bill_actual']
) * 100
proj_agg['acc_model'] = (
    1 - abs(proj_agg['bill_model'] - proj_agg['bill_actual'])
        / proj_agg['bill_actual']
) * 100

arch_acc = proj_agg.groupby('Archetype')[['acc_classic','acc_model']].mean()
arch_acc = arch_acc.reindex([a for a in ARCH_ORDER if a in arch_acc.index])

fig, ax = plt.subplots(figsize=(10, 5))
x3 = np.arange(len(arch_acc))
bars_c3 = ax.bar(x3 - w/2, arch_acc['acc_classic'], w,
                 label='Classic', color='#d62728', alpha=0.82)
bars_m3 = ax.bar(x3 + w/2, arch_acc['acc_model'],   w,
                 label='Model',   color='#1f77b4', alpha=0.82)
ax.set_xticks(x3)
ax.set_xticklabels([ARCH_LABELS.get(a, a) for a in arch_acc.index], fontsize=10)
ax.set_ylabel('Budget Accuracy (%) — higher is better')
ax.set_title('Project-Level Budget Accuracy — Classic vs Model\n'
             'Stage 1 | Billing in USD (cost + 10% markup) | aggregated per project',
             fontsize=11)
ax.axhline(100, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax.set_ylim(max(0, ax.get_ylim()[0] - 2), 105)
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bar in list(bars_c3) + list(bars_m3):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'budget_accuracy_s1.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nBudget accuracy by archetype (project level, billing USD):")
print(f"  {'Archetype':<14} {'Classic':>10} {'Model':>10} {'Delta':>8}")
print(f"  {'─'*46}")
for arch, row in arch_acc.iterrows():
    delta = row['acc_model'] - row['acc_classic']
    print(f"  {arch:<14} {row['acc_classic']:>9.2f}%  {row['acc_model']:>9.2f}%  {delta:>+7.2f}pp")

# ── Chart 3: Per-project USD improvement scatter ───────────────────────────
proj_agg['bill_err_classic'] = abs(proj_agg['bill_classic'] - proj_agg['bill_actual'])
proj_agg['bill_err_model']   = abs(proj_agg['bill_model']   - proj_agg['bill_actual'])
proj_agg['improvement_usd']  = proj_agg['bill_err_classic'] - proj_agg['bill_err_model']

arch_colors = {
    'mobile_tiny': '#4e9af1', 'mobile_small': '#2196f3',
    'indie': '#ff9800', 'mid': '#8bc34a',
    'aaa': '#e91e63', 'mega': '#9c27b0',
}

fig, ax = plt.subplots(figsize=(11, 6))

for arch in ARCH_ORDER:
    subset = proj_agg[proj_agg['Archetype'] == arch]
    if subset.empty:
        continue
    ax.scatter(
        subset['bill_actual'] / 1000,
        subset['improvement_usd'] / 1000,
        c=arch_colors.get(arch, '#888'),
        s=subset['bill_actual'] / subset['bill_actual'].max() * 300 + 40,
        alpha=0.80,
        edgecolors='white', linewidths=0.5,
        label=ARCH_LABELS.get(arch, arch).replace('\n', ' '),
        zorder=3,
    )

# Label notable projects
for _, row in proj_agg.iterrows():
    if abs(row['improvement_usd']) > proj_agg['improvement_usd'].std():
        ax.annotate(
            row['ProjectID'],
            (row['bill_actual']/1000, row['improvement_usd']/1000),
            textcoords='offset points', xytext=(6, 3),
            fontsize=7, color='#444',
        )

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Total Project Billing — Actual (USD thousands)', fontsize=10)
ax.set_ylabel('Budget Deviation Improvement (USD thousands)\nClassic error − Model error  |  positive = model is better', fontsize=9)
ax.set_title('Per-Project Budget Accuracy Improvement — Stage 1\n'
             'Each dot is one project. Size = total billing. '
             'Above zero line = model reduces budget deviation.',
             fontsize=11)
ax.legend(title='Project type', fontsize=8, title_fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'project_improvement_s1.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPer-project USD improvement (top 10 by improvement):")
top = proj_agg.sort_values('improvement_usd', ascending=False).head(10)
print(f"  {'Project':<10} {'Arch':<14} {'Classic Err':>12} {'Model Err':>12} {'Improvement':>13}")
print(f"  {'─'*65}")
for _, r in top.iterrows():
    print(f"  {r['ProjectID']:<10} {r['Archetype']:<14} "
          f"${r['bill_err_classic']:>10,.0f}  ${r['bill_err_model']:>10,.0f}  "
          f"${r['improvement_usd']:>11,.0f}")

# Print EUR impact summary
print(f"\nEUR impact by archetype (mean absolute error per session):")
print(f"  {'Archetype':<14} {'Classic':>12} {'Model':>12} {'Saving':>10}")
print(f"  {'─'*52}")
for arch, row in arch_eur.iterrows():
    saving = row['abs_err_classic'] - row['abs_err_model']
    c = row['abs_err_classic']
    m = row['abs_err_model']
    print(f"  {arch:<14} €{c:>9,.2f}  €{m:>9,.2f}  €{saving:>7,.2f}")

# SHAP
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, feature_names=FEATURES,
                  show=False, plot_size=None)
plt.title("SHAP — Stage 1 Model", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'shap_s1.png', dpi=150, bbox_inches='tight')
plt.close()

# Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (y_hat, label, color) in zip(axes, [
    (y_classic.values, f"Classic  MAPE={mape_classic:.1f}%", "#d62728"),
    (y_pred,           f"XGBoost  MAPE={mape_model:.1f}%",  "#1f77b4"),
]):
    ax.scatter(y_test, y_hat, alpha=0.35, s=18, color=color)
    lim = max(y_test.max(), np.array(y_hat).max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1)
    ax.set_xlabel("Actual Cost (EUR)")
    ax.set_ylabel("Predicted Cost (EUR)")
    ax.set_title(label)
    ax.ticklabel_format(style='plain', axis='both')
plt.suptitle("Stage 1 — Predicted vs Actual", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'pred_actual_s1.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved to {MODEL_DIR}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
joblib.dump(model,   MODEL_DIR / 'model_s1.pkl')
joblib.dump(encoder, MODEL_DIR / 'encoder_s1.pkl')

metadata = {
    'stage':           1,
    'features':        FEATURES,
    'numeric_features':  NUMERICS,
    'categorical_features': CATEGORICALS,
    'encoder_categories': {
        col: list(cats)
        for col, cats in zip(CATEGORICALS, encoder.categories_)
    },
    'metrics': {
        'mape_classic':     round(mape_classic, 2),
        'mape_model_test':  round(mape_model,   2),
        'mape_model_cv':    round(mape_cv,       2),
        'r2':               round(r2_model,      3),
        'improvement_pp':   round(mape_classic - mape_model, 2),
    },
    'training_samples': int(len(X_train)),
    'test_samples':     int(len(X_test)),
}
with open(MODEL_DIR / 'metadata_s1.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ model_s1.pkl, encoder_s1.pkl, metadata_s1.json saved.")
