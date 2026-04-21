# =============================================================================
#  VEHICLE NETWORK DATASET — COMPLETE ML PROJECT
#  Steps: EDA → Preprocessing → Classification → Regression → Clustering →
#         Prediction on New Data
#  Dataset: 3000 rows × 23 columns (VANET vehicular network simulation)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Colour palette used throughout all plots
PURPLE  = '#534AB7'
TEAL    = '#1D9E75'
CORAL   = '#D85A30'
AMBER   = '#EF9F27'
GRAY    = '#888780'
COLORS  = [PURPLE, TEAL, CORAL, AMBER, GRAY]

print("=" * 65)
print("  VEHICLE NETWORK ML PROJECT — FULL ANALYSIS")
print("=" * 65)


# =============================================================================
# STEP 1 — LOAD & UNDERSTAND THE DATASET
# =============================================================================
print("\n>>> STEP 1: Loading dataset ...\n")

df = pd.read_csv("Vehicle_dataset.csv")

print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Column names:\n  {list(df.columns)}")
print(f"\n  Data types:\n{df.dtypes.to_string()}")
print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
print(f"\n  Basic statistics:\n{df.describe().round(2).to_string()}")

# Unique values in categorical columns
cat_cols = ['mobility_pattern', 'RSU_id', 'route_id',
            'routing_stability', 'optimal_action']
print("\n  Unique values in categorical columns:")
for col in cat_cols:
    print(f"    {col}: {df[col].unique().tolist()}")


# =============================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n>>> STEP 2: Exploratory Data Analysis (EDA) ...\n")

# --- Figure 1: Distributions of key numeric features ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("STEP 2 — Distribution of Key Features", fontsize=14, fontweight='bold')

num_features = ['speed', 'link_quality_dB', 'throughput_Mbps',
                'end_to_end_delay_ms', 'reward_value', 'interference_dB']
for ax, feat, color in zip(axes.flat, num_features, COLORS * 2):
    ax.hist(df[feat], bins=35, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    mean_val = df[feat].mean()
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.2,
               label=f'Mean: {mean_val:.1f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig1_distributions.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig1_distributions.png")

# --- Figure 2: Categorical column counts ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("STEP 2 — Categorical Column Distributions", fontsize=14, fontweight='bold')

for ax, col, palette in zip(axes,
    ['mobility_pattern', 'optimal_action', 'routing_stability'],
    [COLORS[:3], [TEAL, CORAL], [PURPLE, AMBER]]):
    counts = df[col].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=palette[:len(counts)], edgecolor='white')
    ax.set_title(col, fontsize=11)
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(val), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("fig2_categoricals.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig2_categoricals.png")

# --- Figure 3: Correlation heatmap ---
plt.figure(figsize=(13, 9))
num_df = df.select_dtypes(include='number')
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.1f',
            cmap='coolwarm', center=0, linewidths=0.4,
            annot_kws={'size': 8})
plt.title("STEP 2 — Correlation Heatmap (lower triangle)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig3_correlation.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig3_correlation.png")

# --- Figure 4: Boxplots — feature vs mobility_pattern ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("STEP 2 — Feature vs Mobility Pattern", fontsize=13, fontweight='bold')
for ax, feat in zip(axes, ['speed', 'throughput_Mbps', 'reward_value']):
    df.boxplot(column=feat, by='mobility_pattern', ax=ax,
               boxprops=dict(color=PURPLE),
               medianprops=dict(color=CORAL, linewidth=2))
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("Mobility Pattern")
plt.suptitle("")
plt.tight_layout()
plt.savefig("fig4_boxplots.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig4_boxplots.png")

print("\n  EDA Summary:")
print(f"    - 3 mobility patterns: {df['mobility_pattern'].value_counts().to_dict()}")
print(f"    - Optimal action split: {df['optimal_action'].value_counts().to_dict()}")
print(f"    - Routing stability: {df['routing_stability'].value_counts().to_dict()}")
print(f"    - Highest corr with reward_value: "
      f"{corr['reward_value'].drop('reward_value').abs().idxmax()}")


# =============================================================================
# STEP 3 — DATA PREPROCESSING
# =============================================================================
print("\n>>> STEP 3: Data Preprocessing ...\n")

# --- Label encode all categorical columns ---
le = LabelEncoder()
enc_map = {}
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col])
    enc_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  Encoded '{col}': {enc_map[col]}")

# --- Define feature matrix and targets ---
feature_cols = [
    'x_position', 'y_position', 'speed', 'vehicle_density',
    'link_quality_dB', 'bandwidth_MHz', 'URLLC_flag',
    'interference_dB', 'tx_power_mW', 'rx_power_mW',
    'residual_energy_J', 'energy_consumed_J',
    'hop_count', 'end_to_end_delay_ms', 'throughput_Mbps',
    'mobility_pattern_enc', 'RSU_id_enc'
]

X = df[feature_cols]
y_action = df['optimal_action_enc']    # 0=next-hop, 1=stay
y_reward = df['reward_value']          # continuous

# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n  Features scaled  : {X_scaled.shape[1]} columns")
print(f"  X_scaled mean    : {X_scaled.mean():.6f}  (should be ~0)")
print(f"  X_scaled std     : {X_scaled.std():.6f}   (should be ~1)")

# --- Train/test split ---
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_action, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled, y_reward, test_size=0.2, random_state=42)

print(f"\n  Train set size   : {len(X_train_c)} rows (80%)")
print(f"  Test set size    : {len(X_test_c)}  rows (20%)")
print(f"  Class balance (train) — next-hop: "
      f"{(y_train_c==0).sum()}, stay: {(y_train_c==1).sum()}")


# =============================================================================
# STEP 4 — CLASSIFICATION: Predict optimal_action
# =============================================================================
print("\n>>> STEP 4: Classification — Predict optimal_action ...\n")

# --- Train three classifiers ---
classifiers = {
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree'       : DecisionTreeClassifier(max_depth=10, random_state=42),
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
}

clf_results = {}
for name, model in classifiers.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred)
    cv  = cross_val_score(model, X_scaled, y_action, cv=5, scoring='accuracy')
    clf_results[name] = {
        'model'   : model,
        'y_pred'  : y_pred,
        'accuracy': acc,
        'cv_mean' : cv.mean(),
        'cv_std'  : cv.std(),
    }
    print(f"  {name}:")
    print(f"    Test Accuracy  : {acc:.4f}")
    print(f"    5-Fold CV      : {cv.mean():.4f} ± {cv.std():.4f}")

best_name = max(clf_results, key=lambda k: clf_results[k]['accuracy'])
best_clf  = clf_results[best_name]['model']
best_pred = clf_results[best_name]['y_pred']
print(f"\n  Best classifier: {best_name}")
print(f"\n  Classification Report ({best_name}):")
print(classification_report(y_test_c, best_pred,
                             target_names=['next-hop', 'stay']))

# --- Figure 5: Confusion matrix + accuracy comparison ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("STEP 4 — Classification Results", fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test_c, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['next-hop', 'stay'],
            yticklabels=['next-hop', 'stay'],
            linewidths=0.5)
axes[0].set_title(f"Confusion Matrix — {best_name}", fontsize=11)
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("Actual Label")

names  = list(clf_results.keys())
accs   = [clf_results[n]['accuracy'] for n in names]
bars   = axes[1].bar(names, accs, color=COLORS[:3], edgecolor='white', width=0.5)
axes[1].set_ylim(0.8, 1.0)
axes[1].set_title("Model Accuracy Comparison", fontsize=11)
axes[1].set_ylabel("Accuracy")
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("fig5_classification.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig5_classification.png")

# --- Figure 6: Feature importance (Random Forest) ---
rf_model = clf_results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances_sorted = importances.sort_values(ascending=True)

plt.figure(figsize=(9, 6))
colors_fi = [PURPLE if v >= importances_sorted.quantile(0.75) else GRAY
             for v in importances_sorted]
importances_sorted.plot(kind='barh', color=colors_fi, edgecolor='white')
plt.title("STEP 4 — Feature Importance (Random Forest)", fontsize=13, fontweight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("fig6_feature_importance.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig6_feature_importance.png")

top3 = importances.nlargest(3).index.tolist()
print(f"\n  Top 3 most important features: {top3}")


# =============================================================================
# STEP 5 — REGRESSION: Predict reward_value
# =============================================================================
print("\n>>> STEP 5: Regression — Predict reward_value ...\n")

regressors = {
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100,
                            learning_rate=0.1, random_state=42),
    'Linear Regression' : LinearRegression(),
}

reg_results = {}
for name, model in regressors.items():
    model.fit(X_train_r, y_train_r)
    y_pred = model.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
    mae  = mean_absolute_error(y_test_r, y_pred)
    r2   = r2_score(y_test_r, y_pred)
    reg_results[name] = {
        'model' : model,
        'y_pred': y_pred,
        'rmse'  : rmse,
        'mae'   : mae,
        'r2'    : r2,
    }
    print(f"  {name}:")
    print(f"    R² Score : {r2:.4f}")
    print(f"    RMSE     : {rmse:.4f}")
    print(f"    MAE      : {mae:.4f}")

best_reg_name  = max(reg_results, key=lambda k: reg_results[k]['r2'])
best_reg_model = reg_results[best_reg_name]['model']
best_reg_pred  = reg_results[best_reg_name]['y_pred']
print(f"\n  Best regressor: {best_reg_name}")

# --- Figure 7: Actual vs Predicted + Residuals ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("STEP 5 — Regression Results", fontsize=14, fontweight='bold')

axes[0].scatter(y_test_r, best_reg_pred, alpha=0.25, s=14, color=PURPLE)
lo, hi = y_test_r.min(), y_test_r.max()
axes[0].plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect fit')
axes[0].set_xlabel("Actual Reward Value")
axes[0].set_ylabel("Predicted Reward Value")
axes[0].set_title(f"Actual vs Predicted — {best_reg_name}", fontsize=11)
axes[0].legend()
axes[0].text(0.05, 0.93,
             f"R² = {reg_results[best_reg_name]['r2']:.4f}\n"
             f"RMSE = {reg_results[best_reg_name]['rmse']:.4f}",
             transform=axes[0].transAxes, fontsize=9,
             bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round'))

residuals = np.array(y_test_r) - best_reg_pred
axes[1].hist(residuals, bins=45, color=TEAL, edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero error')
axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residuals Distribution", fontsize=11)
axes[1].legend()

plt.tight_layout()
plt.savefig("fig7_regression.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig7_regression.png")


# =============================================================================
# STEP 6 — CLUSTERING: K-Means + PCA visualisation
# =============================================================================
print("\n>>> STEP 6: Clustering — K-Means on vehicle behaviour ...\n")

# --- Elbow method to find best k ---
inertia = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# --- Apply k=3 ---
BEST_K = 3
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = df.groupby('cluster')[[
    'speed', 'throughput_Mbps', 'link_quality_dB',
    'reward_value', 'vehicle_density', 'end_to_end_delay_ms'
]].mean().round(2)
print(f"  Cluster summary (k={BEST_K}):\n{cluster_summary.to_string()}")

# Mobility pattern distribution per cluster
print("\n  Mobility pattern per cluster:")
print(pd.crosstab(df['cluster'], df['mobility_pattern']).to_string())

# --- PCA for 2-D visualisation ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_.sum()
print(f"\n  PCA variance explained (2 components): {explained:.2%}")

# --- Figure 8: Elbow + PCA clusters ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("STEP 6 — Clustering Analysis", fontsize=14, fontweight='bold')

axes[0].plot(k_range, inertia, 'o-', color=PURPLE, linewidth=2, markersize=6)
axes[0].axvline(BEST_K, color=CORAL, linestyle='--', linewidth=1.5,
                label=f'Chosen k={BEST_K}')
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-cluster SSE)")
axes[0].set_title("Elbow Method", fontsize=11)
axes[0].legend()

cluster_colors = [PURPLE, TEAL, CORAL]
for c in range(BEST_K):
    mask = df['cluster'] == c
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=cluster_colors[c], alpha=0.35, s=12,
                    label=f'Cluster {c}')
axes[1].set_title(f"PCA 2-D Projection ({explained:.1%} variance)", fontsize=11)
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].legend(markerscale=2)

plt.tight_layout()
plt.savefig("fig8_clustering.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig8_clustering.png")

# --- Figure 9: Cluster profile radar-style bar chart ---
fig, axes = plt.subplots(1, BEST_K, figsize=(13, 4), sharey=False)
fig.suptitle("STEP 6 — Cluster Profiles", fontsize=14, fontweight='bold')
profile_cols = ['speed','throughput_Mbps','link_quality_dB','reward_value']
for c, ax in enumerate(axes):
    vals = cluster_summary.loc[c, profile_cols]
    ax.bar(profile_cols, vals, color=cluster_colors[c], edgecolor='white', alpha=0.85)
    ax.set_title(f"Cluster {c}", fontsize=11)
    ax.set_xticklabels(profile_cols, rotation=20, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig("fig9_cluster_profiles.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: fig9_cluster_profiles.png")


# =============================================================================
# STEP 7 — PREDICT ON 3 NEW UNSEEN VEHICLES
# =============================================================================
print("\n>>> STEP 7: Predicting on New Vehicles ...\n")

# Three brand-new vehicles not present in the training data
new_vehicles_raw = pd.DataFrame([
    {   # Vehicle A — Urban, good signal, URLLC active
        'x_position':45.0, 'y_position':60.0, 'speed':18.5,
        'vehicle_density':70, 'link_quality_dB':32.0,
        'bandwidth_MHz':20, 'URLLC_flag':1,
        'interference_dB':6.0, 'tx_power_mW':130.0, 'rx_power_mW':90.0,
        'residual_energy_J':800.0, 'energy_consumed_J':5.0,
        'hop_count':4, 'end_to_end_delay_ms':22.0, 'throughput_Mbps':30.0,
        'mobility_pattern_enc': enc_map['mobility_pattern']['urban'],
        'RSU_id_enc':          enc_map['RSU_id']['RSU_2'],
    },
    {   # Vehicle B — Rural, weak signal, high delay
        'x_position':10.0, 'y_position':20.0, 'speed':8.0,
        'vehicle_density':25, 'link_quality_dB':17.0,
        'bandwidth_MHz':10, 'URLLC_flag':0,
        'interference_dB':12.5, 'tx_power_mW':170.0, 'rx_power_mW':58.0,
        'residual_energy_J':550.0, 'energy_consumed_J':8.5,
        'hop_count':8, 'end_to_end_delay_ms':46.0, 'throughput_Mbps':9.0,
        'mobility_pattern_enc': enc_map['mobility_pattern']['rural'],
        'RSU_id_enc':           enc_map['RSU_id']['RSU_5'],
    },
    {   # Vehicle C — Highway, excellent signal, high speed
        'x_position':80.0, 'y_position':85.0, 'speed':28.0,
        'vehicle_density':90, 'link_quality_dB':38.5,
        'bandwidth_MHz':25, 'URLLC_flag':1,
        'interference_dB':3.0, 'tx_power_mW':100.0, 'rx_power_mW':116.0,
        'residual_energy_J':950.0, 'energy_consumed_J':3.0,
        'hop_count':3, 'end_to_end_delay_ms':9.0, 'throughput_Mbps':47.0,
        'mobility_pattern_enc': enc_map['mobility_pattern']['highway'],
        'RSU_id_enc':           enc_map['RSU_id']['RSU_1'],
    },
])

# Scale using the SAME scaler fitted on training data (IMPORTANT — no fit_transform)
new_scaled = scaler.transform(new_vehicles_raw[feature_cols])

# --- Predictions ---
action_preds   = best_clf.predict(new_scaled)
action_proba   = best_clf.predict_proba(new_scaled)
reward_preds   = best_reg_model.predict(new_scaled)
cluster_preds  = kmeans.predict(new_scaled)

action_labels  = {0: 'next-hop', 1: 'stay'}
vehicle_labels = [
    'Vehicle A — Urban  (good signal)',
    'Vehicle B — Rural  (weak signal)',
    'Vehicle C — Highway (excellent signal)',
]

print("  " + "=" * 60)
print("  PREDICTION RESULTS")
print("  " + "=" * 60)
for i, label in enumerate(vehicle_labels):
    action      = action_labels[action_preds[i]]
    confidence  = action_proba[i].max() * 100
    reward      = reward_preds[i]
    cluster     = cluster_preds[i]
    print(f"\n  {label}")
    print(f"    Optimal action   : {action}")
    print(f"    Confidence       : {confidence:.1f}%")
    print(f"    Expected reward  : {reward:.3f}")
    print(f"    Behaviour cluster: {cluster}")

# --- Figure 10: Prediction summary bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("STEP 7 — New Vehicle Predictions", fontsize=14, fontweight='bold')

short_labels = ['A (Urban)', 'B (Rural)', 'C (Highway)']
bar_colors   = [TEAL if action_labels[a] == 'stay' else CORAL
                for a in action_preds]
axes[0].bar(short_labels, [action_proba[i].max() * 100 for i in range(3)],
            color=bar_colors, edgecolor='white', width=0.5)
axes[0].set_ylim(0, 110)
axes[0].set_ylabel("Confidence (%)")
axes[0].set_title("Prediction Confidence\n(green=stay, red=next-hop)", fontsize=10)
for j, (sl, ap) in enumerate(zip(short_labels, action_preds)):
    axes[0].text(j, action_proba[j].max() * 100 + 2,
                 action_labels[ap], ha='center', fontsize=9, fontweight='bold')

axes[1].bar(short_labels, reward_preds,
            color=[PURPLE, GRAY, TEAL], edgecolor='white', width=0.5)
axes[1].set_ylabel("Predicted Reward Value")
axes[1].set_title("Predicted Reward Value", fontsize=10)
for j, rv in enumerate(reward_preds):
    axes[1].text(j, rv + 0.05, f'{rv:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("fig10_predictions.png", dpi=120, bbox_inches='tight')
plt.show()
print("\n  Saved: fig10_predictions.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  PROJECT SUMMARY")
print("=" * 65)
print(f"\n  Dataset         : 3000 rows × 23 columns — no missing values")
print(f"\n  CLASSIFICATION (predict optimal_action)")
for name, res in clf_results.items():
    print(f"    {name:<22}: Accuracy = {res['accuracy']:.4f}  |  CV = {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
print(f"\n  REGRESSION (predict reward_value)")
for name, res in reg_results.items():
    print(f"    {name:<22}: R² = {res['r2']:.4f}  |  RMSE = {res['rmse']:.4f}  |  MAE = {res['mae']:.4f}")
print(f"\n  CLUSTERING")
print(f"    K-Means (k=3)         : PCA variance explained = {explained:.2%}")
print(f"    Cluster sizes         : {df['cluster'].value_counts().sort_index().to_dict()}")
print(f"\n  Top 3 features for routing decision: {top3}")
print(f"\n  Figures saved: fig1 through fig10 (.png)")
print("\n" + "=" * 65)
print("  Analysis complete.")
print("=" * 65)