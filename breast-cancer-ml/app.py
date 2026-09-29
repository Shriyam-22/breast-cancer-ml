"""
Streamlit web app for the breast cancer diagnosis model.
Run:  streamlit run app.py   (train first with: python src/train.py)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Breast Cancer Diagnosis Predictor", page_icon="🩺", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "model.joblib")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    return model, meta


model, meta = load()
features = meta["feature_names"]

st.title("🩺 Breast Cancer Diagnosis Predictor")
st.caption("Predicts whether a breast mass is benign or malignant from cell-nucleus measurements "
           "(Wisconsin Diagnostic dataset). For educational purposes only — not a medical device.")

tab1, tab2, tab3 = st.tabs(["Single prediction", "Batch (CSV)", "Model performance"])

with tab1:
    st.sidebar.header("Cell nucleus measurements")
    values = {}
    for f in features:
        values[f] = st.sidebar.slider(
            f, float(meta["feature_min"][f]), float(meta["feature_max"][f]),
            float(meta["feature_means"][f]))
    X = pd.DataFrame([values])
    prob = float(model.predict_proba(X)[0, 1])
    label = "Malignant" if prob >= 0.5 else "Benign"
    c1, c2 = st.columns(2)
    c1.metric("Prediction", label)
    c2.metric("Malignant probability", f"{prob:.1%}")
    st.progress(prob)
    st.info("Adjust the sliders in the sidebar to see how the prediction changes.")

with tab2:
    up = st.file_uploader("Upload a CSV with the 30 feature columns", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        else:
            probs = model.predict_proba(df[features])[:, 1]
            df["malignant_probability"] = probs.round(4)
            df["prediction"] = ["Malignant" if p >= 0.5 else "Benign" for p in probs]
            st.dataframe(df[["prediction", "malignant_probability"]])
            st.download_button("Download results", df.to_csv(index=False), "predictions.csv")

with tab3:
    st.subheader(f"Best model: {meta['best_model']}")
    st.json(meta["test_metrics"])
    fig_dir = ROOT / "reports" / "figures"
    cols = st.columns(2)
    for i, name in enumerate(["confusion_matrix.png", "roc_curves.png",
                              "feature_importance.png", "class_distribution.png"]):
        path = fig_dir / name
        if path.exists():
            cols[i % 2].image(str(path), use_container_width=True)
