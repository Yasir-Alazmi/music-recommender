"""
preprocess.py
-------------
Data loading, cleaning, and feature normalization for the
Music Recommender System.

Dataset: Spotify Songs dataset
  - Download from Kaggle:
    https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset
    OR: https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db
  - Place the CSV as: data/spotify_songs.csv

Expected columns (minimum):
    song_name, artist_name, genre,
    danceability, energy, key, loudness, mode,
    speechiness, acousticness, instrumentalness,
    liveness, valence, tempo, duration_ms, time_signature

Key design decisions:
    • We normalize with StandardScaler (zero-mean, unit-variance) rather than
      MinMaxScaler, because KNN distance is sensitive to outliers and StandardScaler
      handles the heavy-tailed distributions of audio features more robustly.
    • We keep a mapping from scaled-index → song metadata for display.
"""

import logging
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Audio features used for similarity computation ─────────────────────────────
# These are all continuous, meaningful for musical similarity.
# We deliberately exclude: key, mode, time_signature (categorical/ordinal)
AUDIO_FEATURES: List[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# ── Metadata columns kept for display (not used in distance computation) ────────
META_COLUMNS: List[str] = ["song_name", "artist_name", "genre"]

# ── Dataset file name ──────────────────────────────────────────────────────────
DATASET_FILENAME = "spotify_songs.csv"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame) -> dict:
    """
    Flexibly detect column name variants across different Spotify CSV exports.
    Returns a mapping: canonical_name → actual_column_name.
    """
    aliases = {
        "song_name"   : ["song_name", "track_name", "name", "title"],
        "artist_name" : ["artist_name", "artists", "artist", "performer"],
        "genre"       : ["genre", "playlist_genre", "track_genre", "category"],
    }
    mapping = {}
    for canonical, variants in aliases.items():
        for v in variants:
            if v in df.columns:
                mapping[canonical] = v
                break
        if canonical not in mapping:
            mapping[canonical] = None   # column absent — will use placeholder
    return mapping


def load_raw_data(data_dir: str) -> pd.DataFrame:
    """
    Load the Spotify songs CSV, validate required columns, and return
    a cleaned DataFrame.

    Parameters
    ----------
    data_dir : str
        Folder containing spotify_songs.csv.

    Returns
    -------
    pd.DataFrame with columns: song_name, artist_name, genre + AUDIO_FEATURES
    """
    path = os.path.join(data_dir, DATASET_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download from Kaggle and place as data/spotify_songs.csv"
        )

    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path)
    logger.info("Raw shape: %s", df.shape)

    # ── Check audio features are present ──────────────────────────────────────
    missing = [f for f in AUDIO_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required audio feature columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── Detect and normalise metadata columns ──────────────────────────────────
    col_map = _detect_columns(df)
    for canonical, actual in col_map.items():
        if actual and actual != canonical and actual in df.columns:
            df = df.rename(columns={actual: canonical})
        elif not actual:
            df[canonical] = "Unknown"   # placeholder if column absent

    # ── Drop rows missing audio features or song name ─────────────────────────
    required = AUDIO_FEATURES + ["song_name"]
    before   = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    dropped  = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing values.", dropped)

    # ── Remove duplicate songs (keep first occurrence) ─────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["song_name", "artist_name"]).reset_index(drop=True)
    logger.info(
        "Dataset ready — %d unique songs  (removed %d duplicates)",
        len(df), before - len(df),
    )

    # Keep only columns we need
    keep = [c for c in ["song_name", "artist_name", "genre"] + AUDIO_FEATURES
            if c in df.columns]
    return df[keep].copy()


def normalize_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale audio features to zero-mean, unit-variance.

    Why StandardScaler for KNN?
    ---------------------------
    KNN computes Euclidean distance.  Without scaling:
      • loudness range ≈ [-60, 0]  contributes ~3600× more than
      • danceability   range ≈ [0, 1]
    Standardization puts every feature on equal footing.

    Parameters
    ----------
    df     : DataFrame with AUDIO_FEATURES columns
    scaler : if provided, transform only (don't re-fit); used at inference time

    Returns
    -------
    (X_scaled  : np.ndarray shape (n, len(AUDIO_FEATURES)),
     scaler    : fitted StandardScaler)
    """
    X = df[AUDIO_FEATURES].values

    if scaler is None:
        logger.info("Fitting StandardScaler on %d songs …", len(df))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        logger.info("Transforming %d songs with existing scaler …", len(df))
        X_scaled = scaler.transform(X)

    logger.info(
        "Feature matrix shape: %s | Features: %s",
        X_scaled.shape,
        AUDIO_FEATURES,
    )
    return X_scaled, scaler


def prepare_data(data_dir: str) -> dict:
    """
    Full preprocessing pipeline convenience function.

    Returns
    -------
    dict with keys:
        df          : cleaned DataFrame (metadata + audio features)
        X_scaled    : normalized feature matrix (np.ndarray)
        scaler      : fitted StandardScaler
        features    : list of feature names used
    """
    df       = load_raw_data(data_dir)
    X_scaled, scaler = normalize_features(df)

    return {
        "df"      : df,
        "X_scaled": X_scaled,
        "scaler"  : scaler,
        "features": AUDIO_FEATURES,
    }


def find_song_index(df: pd.DataFrame, song_name: str, artist_name: str = None) -> int:
    """
    Locate a song in the DataFrame by name (and optionally artist).

    Parameters
    ----------
    df          : the cleaned songs DataFrame
    song_name   : song title to search for (case-insensitive)
    artist_name : optional artist to disambiguate (case-insensitive)

    Returns
    -------
    Integer row index in df.

    Raises
    ------
    ValueError if song is not found.
    """
    name_lower = song_name.lower().strip()
    mask = df["song_name"].str.lower().str.strip() == name_lower

    if artist_name:
        artist_lower = artist_name.lower().strip()
        mask &= df["artist_name"].str.lower().str.strip() == artist_lower

    matches = df[mask]

    if matches.empty:
        hint = df["song_name"].str.lower().str.contains(name_lower[:5], na=False)
        suggestions = df[hint]["song_name"].head(5).tolist()
        raise ValueError(
            f"Song '{song_name}' not found in dataset.\n"
            f"Did you mean one of: {suggestions}"
        )

    # If multiple matches (e.g. live versions), take the first
    return int(matches.index[0])
