"""
Stage 2 Model — Actor-Level Cost Prediction
Run from project root: python models/train_stage2.py

Context: At Stage 2 you have the actual cast. Classic baseline uses
estimated hours × known actor rate (VIP-aware). The model improves by
learning individual actor efficiency patterns from their session history —
the main signal the classic formula cannot encode.

Key design note:
Actor_Hist_Ratio and Actor_Session_Count are computed from the TRAINING set
only, then mapped to the test set (test actors get training-derived stats,
unseen actors get global mean). This avoids target leakage.
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
DATA_PATH = ROOT / 'data' / 'vo_dataset_stage2.csv'
MODEL_DIR = ROOT / 'models'
TARGET    = 'Cost_Actuals_S2'
RATIO_COL = '_ratio'  # what the model actually predicts

# ── FEATURES ──────────────────────────────────────────────────────────────────
CATEGORICALS = ['Language', 'Vendor']
NUMERICS     = [
    'LS_Words', 'SS_Words', 'HTR_Words', 'STR_Words', 'NoTR_Words',
    'Normalized_Wordcount',
    'Estimated_Hours',
    'Cost_per_Hour',
    'Cost_Forecast_S2',      # strong prior — model learns residual adjustment
    'Is_VIP',
    'Is_Sequel',
    'Actor_Session_Count',   # reliability signal: more history = more confident
    'Actor_Hist_Ratio',      # efficiency fingerprint: actor's historical over/underrun
    'Actor_Hist_Std',        # uncertainty signal: high variance actors less predictable
]
FEATURES = NUMERICS + CATEGORICALS

# ── LOAD & FILTER ─────────────────────────────────────────────────────────────
print("Loading Stage 2 data...")
df = pd.read_csv(DATA_PATH)
df = df[df[TARGET].notna()].copy()
print(f"  {len(df):,} past actor-session rows available")

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
df['Is_VIP']    = df['Is_VIP'].astype(int)
df['Is_Sequel'] = df['Is_Sequel'].astype(int)
df[RATIO_COL]   = df['Cost_Actuals_S2'] / df['Cost_Forecast_S2']

# Efficiency ratio per row (for computing actor history)
df['_actual_ratio'] = df['Actual_Hours'] / df['Estimated_Hours']

# ── TRAIN / TEST SPLIT (before computing actor stats to prevent leakage) ──────
# Split on unique SessionIDs so all actors in a session stay together
sessions     = df['SessionID'].unique().tolist()
train_sess, test_sess = train_test_split(sessions, test_size=0.20, random_state=42)
train_mask   = df['SessionID'].isin(train_sess)
df_train_raw = df[train_mask].copy()
df_test_raw  = df[~train_mask].copy()

print(f"  Train rows: {len(df_train_raw):,}  |  Test rows: {len(df_test_raw):,}")

# ── ACTOR HISTORICAL FEATURES ─────────────────────────────────────────────────
# Computed from TRAINING set only, then mapped to both train and test.
# Test actors get their training-derived stats.
# Actors not seen in training get the global mean (cold-start).

global_mean_ratio = df_train_raw['_actual_ratio'].mean()

actor_stats = (
    df_train_raw.groupby('ActorID')['_actual_ratio']
    .agg(Actor_Hist_Ratio='mean',
         Actor_Hist_Std='std',
         Actor_Session_Count='count')
    .reset_index()
)
actor_stats['Actor_Hist_Std'] = actor_stats['Actor_Hist_Std'].fillna(0)

global_hist_std = actor_stats['Actor_Hist_Std'].mean()

def add_actor_features(df_in, actor_stats, fallback_ratio, fallback_std):
    df_out = df_in.merge(actor_stats, on='ActorID', how='left')
    df_out['Actor_Hist_Ratio']    = df_out['Actor_Hist_Ratio'].fillna(fallback_ratio)
    df_out['Actor_Hist_Std']      = df_out['Actor_Hist_Std'].fillna(fallback_std)
    df_out['Actor_Session_Count'] = df_out['Actor_Session_Count'].fillna(0).astype(int)
    return df_out

df_train = add_actor_features(df_train_raw, actor_stats, global_mean_ratio, global_hist_std)
df_test  = add_actor_features(df_test_raw,  actor_stats, global_mean_ratio, global_hist_std)

# ── ENCODE ────────────────────────────────────────────────────────────────────
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# Fit on training data only
df_train[CATEGORICALS] = encoder.fit_transform(df_train[CATEGORICALS])
df_test[CATEGORICALS]  = encoder.transform(df_test[CATEGORICALS])

X_train = df_train[FEATURES]
y_train = df_train[RATIO_COL]   # predict ratio: actual/forecast
X_test  = df_test[FEATURES]
y_test_ratio  = df_test[RATIO_COL]
y_test_cost   = df_test[TARGET]   # actual cost — for final MAPE evaluation

# ── CLASSIC BASELINE ──────────────────────────────────────────────────────────
y_classic    = df_test_raw['Cost_Forecast_S2']
mape_classic = mean_absolute_percentage_error(y_test_cost, y_classic) * 100
print(f"\nClassic baseline MAPE: {mape_classic:.1f}%")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("\nTraining Stage 2 XGBoost (predicting cost ratio)...")
model = xgb.XGBRegressor(
    n_estimators     = 700,
    learning_rate    = 0.03,
    max_depth        = 5,
    subsample        = 0.80,
    colsample_bytree = 0.75,
    min_child_weight = 5,
    reg_alpha        = 0.10,
    reg_lambda       = 2.0,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test_ratio)],
          verbose=False)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
ratio_pred   = model.predict(X_test)
# Final cost prediction = predicted ratio × classic forecast
cost_pred    = ratio_pred * df_test_raw['Cost_Forecast_S2'].values
mape_model   = mean_absolute_percentage_error(y_test_cost, cost_pred) * 100
r2_model     = r2_score(y_test_cost, cost_pred)
cv_scores    = cross_val_score(model, X_train, y_train,
                               cv=5, scoring='neg_mean_absolute_percentage_error')
mape_cv      = -cv_scores.mean() * 100   # CV is on ratio scale

df_test_eval              = df_test_raw.copy()
df_test_eval['y_pred']    = cost_pred
df_test_eval['y_true']    = y_test_cost.values
df_test_eval['y_classic'] = y_classic.values
df_test_eval['mape_model']   = abs(df_test_eval['y_pred']    - df_test_eval['y_true']) / df_test_eval['y_true'] * 100
df_test_eval['mape_classic'] = abs(df_test_eval['y_classic'] - df_test_eval['y_true']) / df_test_eval['y_true'] * 100

print(f"\n{'─'*52}")
print(f"  Classic MAPE (test)      : {mape_classic:.1f}%")
print(f"  Model MAPE  (test)       : {mape_model:.1f}%")
print(f"  Model MAPE  (5-fold CV)  : {mape_cv:.1f}%")
print(f"  R²                       : {r2_model:.3f}")
print(f"  Improvement              : {mape_classic - mape_model:.1f} pp")
print(f"{'─'*52}")

# Breakdown by language
print(f"\nMAPE by language:")
lang_mape = df_test_eval.groupby('Language')[['mape_classic','mape_model']].mean()
print(f"  {'Language':<24} {'Classic':>8} {'Model':>8}")
print(f"  {'─'*44}")
for lang, row in lang_mape.sort_values('mape_model').iterrows():
    print(f"  {lang:<24} {row['mape_classic']:>7.1f}%  {row['mape_model']:>7.1f}%")

# VIP vs non-VIP
print(f"\nMAPE by VIP status:")
for vip, grp in df_test_eval.groupby('Is_VIP'):
    label = 'VIP' if vip else 'Standard'
    mc = grp['mape_classic'].mean()
    mm = grp['mape_model'].mean()
    print(f"  {label:<10}  Classic={mc:.1f}%  Model={mm:.1f}%  n={len(grp):,}")

# Actor seen vs unseen
df_test_eval['actor_seen'] = df_test_eval['ActorID'].isin(actor_stats['ActorID'])
print(f"\nMAPE — seen vs unseen actors in training:")
for seen, grp in df_test_eval.groupby('actor_seen'):
    label = 'Seen in training' if seen else 'Unseen (cold start)'
    mc = grp['mape_classic'].mean()
    mm = grp['mape_model'].mean()
    print(f"  {label:<22}  Classic={mc:.1f}%  Model={mm:.1f}%  n={len(grp):,}")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

ARCH_ORDER = ['mobile_tiny','mobile_small','indie','mid','aaa','mega']
ARCH_LABELS = {
    'mobile_tiny':  'Mobile\nTiny', 'mobile_small': 'Mobile\nSmall',
    'indie': 'Indie', 'mid': 'Mid', 'aaa': 'AAA', 'mega': 'Mega',
}

# ── Chart 1: MAPE and EUR impact by archetype ──────────────────────────────
arch_mape = df_test_eval.groupby('Archetype')[['mape_classic','mape_model']].mean()
arch_mape = arch_mape.reindex([a for a in ARCH_ORDER if a in arch_mape.index])
df_test_eval['abs_err_classic'] = abs(df_test_eval['y_classic'] - df_test_eval['y_true'])
df_test_eval['abs_err_model']   = abs(df_test_eval['y_pred']    - df_test_eval['y_true'])
arch_eur = df_test_eval.groupby('Archetype')[['abs_err_classic','abs_err_model']].mean()
arch_eur = arch_eur.reindex([a for a in ARCH_ORDER if a in arch_eur.index])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
w = 0.35

ax = axes[0]
x = np.arange(len(arch_mape))
bc = ax.bar(x - w/2, arch_mape['mape_classic'], w, label='Classic', color='#d62728', alpha=0.82)
bm = ax.bar(x + w/2, arch_mape['mape_model'],   w, label='Model',   color='#1f77b4', alpha=0.82)
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABELS.get(a,a) for a in arch_mape.index], fontsize=9)
ax.set_ylabel('MAPE (%)')
ax.set_title('Quoting Error by Project Size — Stage 2', fontsize=11)
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bar in list(bc) + list(bm):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=7.5)

ax2 = axes[1]
x2 = np.arange(len(arch_eur))
bc2 = ax2.bar(x2 - w/2, arch_eur['abs_err_classic'], w, label='Classic', color='#d62728', alpha=0.82)
bm2 = ax2.bar(x2 + w/2, arch_eur['abs_err_model'],   w, label='Model',   color='#1f77b4', alpha=0.82)
ax2.set_xticks(x2)
ax2.set_xticklabels([ARCH_LABELS.get(a,a) for a in arch_eur.index], fontsize=9)
ax2.set_ylabel('Mean Absolute Error (EUR per actor-session)')
ax2.set_title('EUR Quoting Error by Project Size — Stage 2', fontsize=11)
ax2.legend(); ax2.grid(axis='y', alpha=0.3)
for bar in list(bc2) + list(bm2):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'€{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'arch_breakdown_s2.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Chart 2: Budget accuracy by project (billing, aggregated to project) ──
MARKUP_RATE = 0.10
df_test_eval['billing_model'] = cost_pred * (1 + MARKUP_RATE) * df_test_raw['Exchange_Rate'].values

proj_agg = df_test_eval.groupby(['ProjectID','Archetype']).agg(
    bill_classic=('Billing_Forecast_S2', 'sum'),
    bill_actual =('Billing_Actuals_S2',  'sum'),
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
ax.set_xticklabels([ARCH_LABELS.get(a,a) for a in arch_acc.index], fontsize=10)
ax.set_ylabel('Budget Accuracy (%) — higher is better')
ax.set_title('Project-Level Budget Accuracy — Classic vs Model\n'
             'Stage 2 | Billing in USD (cost + 10% markup) | aggregated per project',
             fontsize=11)
ax.axhline(100, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax.set_ylim(max(0, ax.get_ylim()[0] - 2), 105)
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bar in list(bars_c3) + list(bars_m3):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'budget_accuracy_s2.png', dpi=150, bbox_inches='tight')
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
ax.set_title('Per-Project Budget Accuracy Improvement — Stage 2\n'
             'Each dot is one project. Size = total billing. '
             'Above zero line = model reduces budget deviation.',
             fontsize=11)
ax.legend(title='Project type', fontsize=8, title_fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'project_improvement_s2.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPer-project USD improvement (top 10 by improvement):")
top = proj_agg.sort_values('improvement_usd', ascending=False).head(10)
print(f"  {'Project':<10} {'Arch':<14} {'Classic Err':>12} {'Model Err':>12} {'Improvement':>13}")
print(f"  {'─'*65}")
for _, r in top.iterrows():
    print(f"  {r['ProjectID']:<10} {r['Archetype']:<14} "
          f"${r['bill_err_classic']:>10,.0f}  ${r['bill_err_model']:>10,.0f}  "
          f"${r['improvement_usd']:>11,.0f}")

print(f"\nEUR impact by archetype (mean absolute error per actor-session):")
print(f"  {'Archetype':<14} {'Classic':>10} {'Model':>10} {'Saving':>8}")
print(f"  {'─'*46}")
for arch, row in arch_eur.iterrows():
    saving = row['abs_err_classic'] - row['abs_err_model']
    c = row['abs_err_classic']
    m = row['abs_err_model']
    print(f"  {arch:<14} €{c:>7,.2f}  €{m:>7,.2f}  €{saving:>5,.2f}")

# SHAP
explainer   = shap.TreeExplainer(model)
shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
shap_values = explainer.shap_values(shap_sample)
fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES,
                  show=False, plot_size=None)
plt.title("SHAP — Stage 2 Model", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'shap_s2.png', dpi=150, bbox_inches='tight')
plt.close()

# Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (y_hat, label, color) in zip(axes, [
    (y_classic.values, f"Classic  MAPE={mape_classic:.1f}%", "#d62728"),
    (cost_pred,        f"XGBoost  MAPE={mape_model:.1f}%",  "#1f77b4"),
]):
    ax.scatter(y_test_cost, y_hat, alpha=0.25, s=10, color=color)
    lim = max(y_test_cost.max(), np.array(y_hat).max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1)
    ax.set_xlabel("Actual Cost (EUR)")
    ax.set_ylabel("Predicted Cost (EUR)")
    ax.set_title(label)
    ax.ticklabel_format(style='plain', axis='both')
plt.suptitle("Stage 2 — Predicted vs Actual", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'pred_actual_s2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved to {MODEL_DIR}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
joblib.dump(model,       MODEL_DIR / 'model_s2.pkl')
joblib.dump(encoder,     MODEL_DIR / 'encoder_s2.pkl')
joblib.dump(actor_stats, MODEL_DIR / 'actor_stats_s2.pkl')

metadata = {
    'stage':              2,
    'features':           FEATURES,
    'numeric_features':   NUMERICS,
    'categorical_features': CATEGORICALS,
    'global_mean_ratio':  round(global_mean_ratio, 4),
    'global_hist_std':    round(global_hist_std,   4),
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
with open(MODEL_DIR / 'metadata_s2.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ model_s2.pkl, encoder_s2.pkl, actor_stats_s2.pkl, metadata_s2.json saved.")
