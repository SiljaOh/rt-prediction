import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

# ======================================================
# 1. PFADE
# ======================================================
TRAIN_PATH = "data/data_for_student-2.csv"
PRED_PATH = "data/for_prediction.csv"
OUT_PATH = "submissionRT.csv"

# ======================================================
# 2. PHYSIKOCHEMISCHE KONSTANTEN
# ======================================================
HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
} #fettliebend = positiv

POS = set("KRH") # positv geladene Aminosäuren
NEG = set("DE") # negativ geladene Aminosäuren
HYDRO_AA = set("AILMFWVY")

# ======================================================
# 3. FEATURE-FUNKTIONEN
# ======================================================
def clean_peptide(seq):
    return re.sub(r"\[UNIMOD:\d+\]", "", seq) #Modifikationen entfernen

def gravy(seq):
    return np.mean([HYDRO.get(a, 0) for a in seq]) # durchschnitts fettliebendkeit

def charge(seq):
    return sum(a in POS for a in seq) - sum(a in NEG for a in seq)  #Ladung für Sequenzen berechnen

def build_features(df):        #features erstellen
    df = df.copy()
    seqs = df["clean_peptide"]

    df["length"] = seqs.str.len()
    df["gravy"] = seqs.apply(gravy)  #fettliebe durschschnitt
    df["charge"] = seqs.apply(charge)
    df["hydrophobic_frac"] = seqs.apply(lambda s: sum(a in HYDRO_AA for a in s) / len(s))  #anteil der fettleibenden
    df["basic_frac"] = seqs.apply(lambda s: sum(a in POS for a in s) / len(s)) #anteil der positiv geladennen aminos
    df["acidic_frac"] = seqs.apply(lambda s: sum(a in NEG for a in s) / len(s)) #anteil der neg geladenen aminos

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    for aa in amino_acids:
        df[f"count_{aa}"] = seqs.str.count(aa)

    return df[
        ["length", "gravy", "charge",
         "hydrophobic_frac", "basic_frac", "acidic_frac"]
        + [f"count_{aa}" for aa in amino_acids]
    ]

# ======================================================
# 4. TRAININGSDATEN LADEN
# ======================================================
train_df = pd.read_csv(TRAIN_PATH)
train_df["clean_peptide"] = train_df["peptide_sequence"].apply(clean_peptide)

X = build_features(train_df) #input
y = train_df["indexed_retention_time"]   # <<< iRT als Zielvariable

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 5. iRT-MODELL (ExtraTrees)
# ======================================================
irt_model = ExtraTreesRegressor(
    n_estimators=600,
    max_depth=30,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

irt_model.fit(X_train, y_train)

val_pred = irt_model.predict(X_val)
irt_mae = mean_absolute_error(y_val, val_pred)

print(f"\n📉 iRT VALIDATION MAE: {irt_mae:.3f}\n")

# ======================================================
# 6. LINEARES MAPPING: iRT → RT (pro raw_file)
# ======================================================
train_df["pred_iRT"] = irt_model.predict(X)

rt_models = {} #dictionary für Formeln
for rf, g in train_df.groupby("raw_file"):
    lr = LinearRegression()
    lr.fit(g[["pred_iRT"]], g["retention_time"])
    rt_models[rf] = lr

# ======================================================
# 7. PREDICTION-DATEN
# ======================================================
pred_df = pd.read_csv(PRED_PATH)
pred_df["clean_peptide"] = pred_df["peptide_sequence"].apply(clean_peptide)

X_pred = build_features(pred_df)
pred_df["pred_iRT"] = irt_model.predict(X_pred)

# ======================================================
# 8. VEKTORISIERTES iRT → RT (OHNE WARNUNGEN)
# ======================================================
pred_df["retention_time"] = np.nan

for rf, lr in rt_models.items():
    mask = pred_df["raw_file"] == rf
    if mask.any():
        X_tmp = pred_df.loc[mask, ["pred_iRT"]]
        pred_df.loc[mask, "retention_time"] = lr.predict(X_tmp)

# Fallback (falls raw_file unbekannt)
pred_df["retention_time"] = (
    pred_df["retention_time"]
    .fillna(pred_df["pred_iRT"])
    .clip(lower=0)
)

# ======================================================
# 9. SUBMISSION SPEICHERN
# ======================================================
submission = pred_df[["peptide_sequence", "retention_time"]]
submission.to_csv(OUT_PATH, index=False)

print("✅ Submission erfolgreich erstellt")
print(f"📁 {OUT_PATH}")
print(f"📊 Zeilen: {len(submission)}")
print(submission.head())
