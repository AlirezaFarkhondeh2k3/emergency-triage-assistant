import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "incidents.csv"
MODEL_DIR = ROOT / "app" / "models" / "transformer_artifacts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "category"])
    return df


def embed_texts(model: SentenceTransformer, texts):
    # texts: list/Series of strings
    return model.encode(
        texts.tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def train_transformer_classifier(df: pd.DataFrame):
    X = df["text"].astype(str)
    y = df["category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    st_model = SentenceTransformer(MODEL_NAME)

    print("Encoding train texts...")
    X_train_emb = embed_texts(st_model, X_train)
    print("Encoding test texts...")
    X_test_emb = embed_texts(st_model, X_test)

    print("Train embedding shape:", X_train_emb.shape)
    print("Test embedding shape:", X_test_emb.shape)

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("Fitting classifier on embeddings...")
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Transformer Macro F1: {macro_f1:.3f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return clf, macro_f1


def save_artifacts(clf, macro_f1: float):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "transformer_classifier.joblib")

    meta = {
        "macro_f1": macro_f1,
        "sentence_transformer_model": MODEL_NAME,
    }
    with open(MODEL_DIR / "transformer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved transformer artifacts to {MODEL_DIR}")


def main():
    print(f"Loading data from {DATA_PATH}")
    df = load_data()
    print("Data shape:", df.shape)
    print("Category counts:\n", df["category"].value_counts())

    clf, macro_f1 = train_transformer_classifier(df)
    save_artifacts(clf, macro_f1)


if __name__ == "__main__":
    main()
