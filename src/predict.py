"""
Command-line prediction using the trained model.

Examples:
  python src/predict.py --sample 0          # predict on a row from the dataset
  python src/predict.py --csv patients.csv  # predict on a CSV with the 30 feature columns
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]


def load():
    model = joblib.load(ROOT / "models" / "model.joblib")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    return model, meta


def predict_df(model, meta, df):
    df = df[meta["feature_names"]]
    probs = model.predict_proba(df)[:, 1]
    out = df.copy()
    out["malignant_probability"] = probs.round(4)
    out["prediction"] = ["Malignant" if p >= 0.5 else "Benign" for p in probs]
    return out


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, help="Row index from the built-in dataset")
    g.add_argument("--csv", type=str, help="CSV file with the 30 feature columns")
    args = p.parse_args()

    model, meta = load()
    if args.sample is not None:
        data = load_breast_cancer(as_frame=True)
        df = data.data.iloc[[args.sample]]
        actual = "Malignant" if data.target.iloc[args.sample] == 0 else "Benign"
        res = predict_df(model, meta, df)
        print(f"Prediction: {res['prediction'].iloc[0]} "
              f"(malignant probability {res['malignant_probability'].iloc[0]:.2%}) | Actual: {actual}")
    else:
        res = predict_df(model, meta, pd.read_csv(args.csv))
        print(res[["malignant_probability", "prediction"]].to_string())


if __name__ == "__main__":
    main()
