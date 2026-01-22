import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')




TRAIN_PATH = "data/data_for_student-2.csv"
PRED_PATH = "data/for_prediction.csv"
OUT_PATH = "submissionRT.csv"


# Deine Feature-Funktionen (kopiert)
HYDRO = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
         "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3}
POS = set("KRH")
NEG = set("DE")
HYDRO_AA = set("AILMFWVY")

def clean_peptide(seq):
    return re.sub(r"\[UNIMOD:\d+\]", "", seq)

def gravy(seq):
    return np.mean([HYDRO.get(a, 0) for a in seq])

def charge(seq):
    return sum(a in POS for a in seq) - sum(a in NEG for a in seq)

def build_features(df):
    df = df.copy()
    seqs = df["clean_peptide"]
    df["length"] = seqs.str.len()
    df["gravy"] = seqs.apply(gravy)
    df["charge"] = seqs.apply(charge)
    df["hydrophobic_frac"] = seqs.apply(lambda s: sum(a in HYDRO_AA for a in s) / len(s))
    df["basic_frac"] = seqs.apply(lambda s: sum(a in POS for a in s) / len(s))
    df["acidic_frac"] = seqs.apply(lambda s: sum(a in NEG for a in s) / len(s))
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    for aa in amino_acids:
        df[f"count_{aa}"] = seqs.str.count(aa)
    return df[["length", "gravy", "charge", "hydrophobic_frac", "basic_frac", "acidic_frac"] +
              [f"count_{aa}" for aa in amino_acids]]

# 2. Trainiere Modell
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
train_df["clean_peptide"] = train_df["peptide_sequence"].apply(clean_peptide)
X = build_features(train_df)
y = train_df["indexed_retention_time"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import ExtraTreesRegressor
irt_model = ExtraTreesRegressor(n_estimators=600, max_depth=30, min_samples_leaf=2, random_state=42, n_jobs=-1)
irt_model.fit(X_train, y_train)

val_pred = irt_model.predict(X_val)
mae = mean_absolute_error(y_val, val_pred)
r2 = r2_score(y_val, val_pred)

print(f"Validation MAE: {mae:.3f}, R²: {r2:.3f}")

# 3. PLOTS ERSTELLEN (4 wunderschöne für Report!)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Model Performance - Retention Time Prediction', fontsize=16, fontweight='bold')

# PLOT 1: Predicted vs Actual iRT (PERFEKT für Report!)
axes[0,0].scatter(y_val, val_pred, alpha=0.6, s=20, color='steelblue', edgecolors='white', linewidth=0.5)
axes[0,0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0,0].set_xlabel('True iRT')
axes[0,0].set_ylabel('Predicted iRT')
axes[0,0].set_title(f'Predicted vs Actual iRT\nMAE={mae:.3f}, R²={r2:.3f}')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# PLOT 2: Feature Importance (Top 10)
importances = irt_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=True).tail(10)

sns.barplot(data=importance_df, y='feature', x='importance', ax=axes[0,1], palette='viridis')
axes[0,1].set_title('Top 10 Feature Importances')
axes[0,1].set_xlabel('Importance')

# PLOT 3: Residuals Plot
residuals = y_val - val_pred
axes[1,0].scatter(val_pred, residuals, alpha=0.6, s=20)
axes[1,0].axhline(y=0, color='r', linestyle='--')
axes[1,0].set_xlabel('Predicted iRT')
axes[1,0].set_ylabel('Residuals')
axes[1,0].set_title('Residuals Plot\n(MAE=2.783)')
axes[1,0].grid(True, alpha=0.3)

# PLOT 4: GRAVY vs iRT (physikalisch sinnvoll!)
train_df['clean_peptide'] = train_df['peptide_sequence'].apply(clean_peptide)
train_df['gravy'] = train_df['clean_peptide'].apply(gravy)
sns.scatterplot(data=train_df.sample(5000), x='gravy', y='indexed_retention_time', ax=axes[1,1], alpha=0.6)
axes[1,1].set_xlabel('GRAVY Score')
axes[1,1].set_ylabel('True iRT')
axes[1,1].set_title('GRAVY vs True iRT\n(Higher GRAVY → Later Elution)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('retention_time_plots_report.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ PLOTS GESPEICHERT: retention_time_plots_report.png")
print("📸 Kopiere in LaTeX mit: \\includegraphics[width=\\textwidth]{retention_time_plots_report.png}")

# 5. BONUS: Feature Correlation Heatmap (optional)
plt.figure(figsize=(12,8))
corr_features = X[['length', 'gravy', 'charge', 'hydrophobic_frac']].corr()
sns.heatmap(corr_features, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()
