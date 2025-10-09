"""
Preprocessing script for Fake Amazon Review Detection.
Handles automatic encoding detection and gzip-compressed CSV files.
Generates TF-IDF, PCA features, and label encodings, and saves transformers in models/.
"""

import os
import re
import gzip
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from utils import save_pickle


def clean_text(s: str) -> str:
    """Simple text cleaner: lowercasing, removing punctuation, multiple spaces."""
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = s.lower()
    s = " ".join(s.split())
    return s


def load_csv_safe(csv_path: str) -> pd.DataFrame:
    """Try reading CSV with multiple encodings and gzip support."""
    encodings = ["utf-8", "ISO-8859-1", "latin1"]

    # Check if file might be gzipped
    if csv_path.endswith(".gz"):
        for enc in encodings:
            try:
                with gzip.open(csv_path, "rt", encoding=enc, errors="ignore") as f:
                    df = pd.read_csv(f, on_bad_lines="skip")
                print(f"Loaded compressed CSV using encoding={enc}")
                return df
            except Exception:
                continue
    else:
        for enc in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=enc, on_bad_lines="skip")
                print(f"Loaded CSV using encoding={enc}")
                return df
            except Exception:
                continue

    raise ValueError("Unable to read CSV file with common encodings or compression formats.")


def build_features(csv_path: str, max_tfidf: int = 100, pca_components: int = 6):
    """Builds TF-IDF, encodings, PCA features, and labels."""
    df = load_csv_safe(csv_path)
    required_cols = ["REVIEW_TEXT", "RATING", "VERIFIED_PURCHASE", "PRODUCT_CATEGORY"]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    df = df.dropna(subset=required_cols)
    df["cleaned_review"] = df["REVIEW_TEXT"].apply(clean_text)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_tfidf)
    X_text = tfidf.fit_transform(df["cleaned_review"]).toarray()
    save_pickle(tfidf, "tfidf.sav")

    # Label encoders
    le_verified = LabelEncoder()
    ver = le_verified.fit_transform(df["VERIFIED_PURCHASE"].astype(str))
    save_pickle(le_verified, "le_verified.sav")

    le_cat = LabelEncoder()
    cat = le_cat.fit_transform(df["PRODUCT_CATEGORY"].astype(str))
    save_pickle(le_cat, "le_category.sav")

    rating = df["RATING"].values.reshape(-1, 1).astype(float)
    X = np.hstack([rating, ver.reshape(-1, 1), cat.reshape(-1, 1), X_text])

    # PCA
    pca_components = min(pca_components, X.shape[1])
    pca = PCA(n_components=pca_components)
    X_reduced = pca.fit_transform(X)
    save_pickle(pca, "pca.sav")

    # Label: fake = rating <= 2
    y = (df["RATING"] <= 2).astype(int).values

    print(f"✅ Preprocessing complete: {X_reduced.shape[0]} samples, {X_reduced.shape[1]} features")
    return X_reduced, y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", required=True)
    args = parser.parse_args()
    X, y = build_features(args.datafile)
    print("Features shape:", X.shape)
