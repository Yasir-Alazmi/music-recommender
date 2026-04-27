"""
explain.py
----------
Improvement #1: Feature-level explainability for recommendations.

Shows WHICH audio features contributed most to the similarity between
the query song and each recommended song, using feature-wise distance.

Usage:
    python -m src.explain --model-dir models/ --song "Blinding Lights" --top-n 5
"""

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocess import find_song_index, AUDIO_FEATURES, normalize_features
from src.train import load_artifacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def feature_contribution(
    query_profile: np.ndarray,
    rec_profile: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Compute the absolute difference per feature between query and recommendation.
    Smaller difference = more similar on that feature.

    Returns a DataFrame sorted by similarity (smallest difference first).
    """
    diffs = np.abs(query_profile - rec_profile)
    total = diffs.sum() + 1e-9
    pct   = (1 - diffs / diffs.max()) * 100   # invert: high = similar

    return pd.DataFrame({
        "feature"     : feature_names,
        "query_value" : np.round(query_profile, 3),
        "rec_value"   : np.round(rec_profile,   3),
        "abs_diff"    : np.round(diffs, 4),
        "similarity%" : np.round(pct,   1),
    }).sort_values("similarity%", ascending=False).reset_index(drop=True)


def explain_recommendation(
    song_name: str,
    artist_name: Optional[str] = None,
    top_n: int = 5,
    model_dir: str = "models",
) -> str:
    """
    Generate a human-readable feature-level explanation for recommendations.

    Returns
    -------
    Formatted string report.
    """
    from src.recommend import MusicRecommender

    rec        = MusicRecommender(model_dir=model_dir)
    results    = rec.recommend(song_name, artist_name=artist_name, top_n=top_n)
    query_idx  = find_song_index(rec.df, song_name, artist_name)
    query_vec  = rec.X_scaled[query_idx]

    lines = [
        "",
        "═" * 65,
        f"  FEATURE EXPLANATION — '{song_name}'" +
        (f" by {artist_name}" if artist_name else ""),
        "═" * 65,
    ]

    for _, row in results.iterrows():
        rec_idx  = find_song_index(rec.df, row["song_name"], row["artist_name"])
        rec_vec  = rec.X_scaled[rec_idx]
        contrib  = feature_contribution(query_vec, rec_vec, AUDIO_FEATURES)

        lines.append(
            f"\n  #{int(row['rank'])}  {row['song_name']} — {row['artist_name']}"
            f"  [{row['similarity']:.1f}% similar]"
        )
        lines.append("  " + "─" * 60)
        lines.append(
            f"  {'Feature':<20} {'Query':>8} {'Rec':>8} {'Match %':>8}"
        )
        lines.append("  " + "─" * 60)
        for _, frow in contrib.head(6).iterrows():
            bar = "█" * int(frow["similarity%"] / 10)
            lines.append(
                f"  {frow['feature']:<20} "
                f"{frow['query_value']:>8.3f} "
                f"{frow['rec_value']:>8.3f} "
                f"{frow['similarity%']:>7.1f}%  {bar}"
            )

    lines.append("\n" + "═" * 65 + "\n")
    return "\n".join(lines)


def _parse_args():
    p = argparse.ArgumentParser(description="Explain music recommendations feature-by-feature.")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--song",      required=True)
    p.add_argument("--artist",    default=None)
    p.add_argument("--top-n",     type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    report = explain_recommendation(
        args.song, artist_name=args.artist,
        top_n=args.top_n, model_dir=args.model_dir,
    )
    print(report)
