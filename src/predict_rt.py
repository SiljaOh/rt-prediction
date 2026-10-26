"""Peptide retention time prediction (winning solution of the TUM RT bonus challenge).

Two-stage hybrid model:
  1. ExtraTrees regressor predicts the instrument-independent indexed retention time (iRT)
     from 26 physicochemical sequence features.
  2. A linear model per instrument run (raw_file) maps predicted iRT to the measured
     retention time (RT = m * iRT + b).

Usage:
    python src/predict_rt.py --train data/train.csv --predict data/for_prediction.csv --out submission.csv
"""

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Kyte-Doolittle hydropathy scale (positive = hydrophobic)
HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
POSITIVE = set("KRH")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWVY")
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def clean_peptide(seq: str) -> str:
    """Remove UNIMOD modification tags, e.g. 'M[UNIMOD:35]' -> 'M'."""
    return re.sub(r"\[UNIMOD:\d+\]", "", seq)


def gravy(seq: str) -> float:
    """Grand average of hydropathy."""
    return float(np.mean([HYDRO.get(a, 0.0) for a in seq]))


def net_charge(seq: str) -> int:
    return sum(a in POSITIVE for a in seq) - sum(a in NEGATIVE for a in seq)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """26 features: 6 physicochemical descriptors + 20 amino acid counts."""
    seqs = df["peptide_sequence"].map(clean_peptide)
    feats = pd.DataFrame(index=df.index)
    feats["length"] = seqs.str.len()
    feats["gravy"] = seqs.map(gravy)
    feats["charge"] = seqs.map(net_charge)
    feats["hydrophobic_frac"] = seqs.map(lambda s: sum(a in HYDROPHOBIC for a in s) / len(s))
    feats["basic_frac"] = seqs.map(lambda s: sum(a in POSITIVE for a in s) / len(s))
    feats["acidic_frac"] = seqs.map(lambda s: sum(a in NEGATIVE for a in s) / len(s))
    for aa in AMINO_ACIDS:
        feats[f"count_{aa}"] = seqs.str.count(aa)
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="CSV with peptide_sequence, indexed_retention_time, retention_time, raw_file")
    ap.add_argument("--predict", required=True, help="CSV with peptide_sequence, raw_file")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    # Stage 1: sequence -> iRT
    train = pd.read_csv(args.train)
    X = build_features(train)
    y = train["indexed_retention_time"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    irt_model = ExtraTreesRegressor(
        n_estimators=600, max_depth=30, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    irt_model.fit(X_tr, y_tr)
    val_pred = irt_model.predict(X_val)
    print(f"iRT validation: MAE = {mean_absolute_error(y_val, val_pred):.3f}, R2 = {r2_score(y_val, val_pred):.3f}")

    # Stage 2: iRT -> RT, one linear calibration per instrument run
    train["pred_iRT"] = irt_model.predict(X)
    calibration = {
        raw_file: LinearRegression().fit(group[["pred_iRT"]], group["retention_time"])
        for raw_file, group in train.groupby("raw_file")
    }

    # Prediction
    pred = pd.read_csv(args.predict)
    pred["pred_iRT"] = irt_model.predict(build_features(pred))
    pred["retention_time"] = np.nan
    for raw_file, lr in calibration.items():
        mask = pred["raw_file"] == raw_file
        if mask.any():
            pred.loc[mask, "retention_time"] = lr.predict(pred.loc[mask, ["pred_iRT"]])

    # Fallback for unseen instrument runs: use the predicted iRT directly
    pred["retention_time"] = pred["retention_time"].fillna(pred["pred_iRT"]).clip(lower=0)

    pred[["peptide_sequence", "retention_time"]].to_csv(args.out, index=False)
    print(f"Saved {len(pred)} predictions to {args.out}")


if __name__ == "__main__":
    main()
