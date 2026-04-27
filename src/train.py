"""
train.py
--------
Fits a K-Nearest Neighbors model on normalized audio features and
persists the model + scaler to disk.

Algorithm spotlight — K-Nearest Neighbors (KNN)
------------------------------------------------
• KNN is a "lazy learner" — it stores all training points and at
  query time finds the K most similar ones by distance.
• No explicit training phase: fit() just memorises the data.
• Distance metric matters:
    - Euclidean: straight-line distance in feature space (default)
    - Cosine:    angle between vectors — better for high-dim data
• We use algorithm='ball_tree' for efficient nearest-neighbour
  lookup on medium-sized datasets (faster than brute force).

Usage:
    python -m src.train --data-dir data/ --model-dir models/
"""

import argparse
import logging
import os
from typing import Tuple

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.preprocess import prepare_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────────────
KNN_N_NEIGHBORS = 20        # fetch more than needed; caller trims to top-N
KNN_METRIC      = "cosine"  # cosine similarity for audio feature vectors
KNN_ALGORITHM   = "brute"   # brute force works best with cosine metric

# ── Artifact filenames ─────────────────────────────────────────────────────────
MODEL_FILE      = "knn_model.joblib"
SCALER_FILE     = "scaler.joblib"
DATAFRAME_FILE  = "songs_df.joblib"    # persist cleaned df for inference


def train_model(X_scaled: np.ndarray) -> NearestNeighbors:
    """
    Fit a NearestNeighbors model on the scaled feature matrix.

    Parameters
    ----------
    X_scaled : np.ndarray — shape (n_songs, n_features)

    Returns
    -------
    Fitted NearestNeighbors instance.
    """
    logger.info(
        "Fitting NearestNeighbors  metric=%s  n_neighbors=%d …",
        KNN_METRIC, KNN_N_NEIGHBORS,
    )
    knn = NearestNeighbors(
        n_neighbors=KNN_N_NEIGHBORS,
        metric=KNN_METRIC,
        algorithm=KNN_ALGORITHM,
        n_jobs=-1,
    )
    knn.fit(X_scaled)
    logger.info("KNN model fitted on %d songs.", X_scaled.shape[0])
    return knn


def save_artifacts(knn, scaler, df, model_dir: str) -> None:
    """Save model, scaler, and song DataFrame to model_dir."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(knn,    os.path.join(model_dir, MODEL_FILE))
    joblib.dump(scaler, os.path.join(model_dir, SCALER_FILE))
    joblib.dump(df,     os.path.join(model_dir, DATAFRAME_FILE))
    logger.info("Artifacts saved → %s", model_dir)


def load_artifacts(model_dir: str) -> Tuple:
    """Load and return (knn, scaler, df) from model_dir."""
    def _load(fname):
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing artifact: {path}")
        return joblib.load(path)

    knn    = _load(MODEL_FILE)
    scaler = _load(SCALER_FILE)
    df     = _load(DATAFRAME_FILE)
    logger.info("Artifacts loaded from %s", model_dir)
    return knn, scaler, df


def _parse_args():
    p = argparse.ArgumentParser(description="Train the Music Recommender KNN model.")
    p.add_argument("--data-dir",  default="data",   help="Folder with spotify_songs.csv")
    p.add_argument("--model-dir", default="models", help="Where to save artifacts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data = prepare_data(args.data_dir)
    knn  = train_model(data["X_scaled"])
    save_artifacts(knn, data["scaler"], data["df"], args.model_dir)
    logger.info("Done! Run recommend.py to get song recommendations.")
