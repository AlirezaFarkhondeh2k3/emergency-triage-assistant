import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "incidents.csv"
MODEL_DIR = ROOT / "app" / "models" / "artifacts"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "category"])
    return df


def train_baseline(df, base_test_size: float = 0.2):
    X = df["text"].astype(str)
    y = df["category"].astype(str)

    n_samples = len(df)
    n_classes = y.nunique()
    min_per_class = y.value_counts().min()

    # Allow CI to run on tiny sample data without stratification errors.
    stratify = y if min_per_class >= 2 else None

    # Ensure test set is large enough for stratification (>= n_classes) but still leaves
    # at least one train sample.
    test_size = max(int(n_samples * base_test_size), 1)
    if stratify is not None:
        test_size = max(test_size, n_classes)
    test_size = min(test_size, n_samples - 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.3f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return vectorizer, clf, macro_f1


def save_artifacts(vectorizer, clf, macro_f1: float):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODEL_DIR / "tfidf.joblib")
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")

    meta = {"macro_f1": macro_f1}
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved artifacts to {MODEL_DIR}")


def main():
    print(f"Loading data from {DATA_PATH}")
    df = load_data()
    print("Data shape:", df.shape)
    print("Category counts:\n", df["category"].value_counts())

    vectorizer, clf, macro_f1 = train_baseline(df)
    save_artifacts(vectorizer, clf, macro_f1)


if __name__ == "__main__":
    main()
