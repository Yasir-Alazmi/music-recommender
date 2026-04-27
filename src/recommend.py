"""
recommend.py
------------
Core recommendation engine.

Given a song name, returns the top-N most similar songs based on
cosine distance in the normalized audio feature space.

Usage (script):
    python -m src.recommend --model-dir models/ --song "Blinding Lights" --artist "The Weeknd" --top-n 10

Usage (module):
    from src.recommend import MusicRecommender
    rec = MusicRecommender(model_dir="models/")
    results = rec.recommend("Blinding Lights", artist_name="The Weeknd", top_n=5)
"""

import argparse
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.preprocess import find_song_index, AUDIO_FEATURES
from src.train import load_artifacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class MusicRecommender:
    """
    Content-based music recommender using pre-trained KNN model.

    Content-based filtering recap
    ------------------------------
    • We recommend songs similar in *audio features* to the query song.
    • No user history needed — works from day 1 (cold-start friendly).
    • Drawback: can't recommend songs outside a user's taste profile
      (filter bubble) — but that's fine for a learning project!
    """

    def __init__(self, model_dir: str = "models") -> None:
        self.knn, self.scaler, self.df = load_artifacts(model_dir)
        # Pre-scale all songs once for fast lookup
        from src.preprocess import normalize_features
        self.X_scaled, _ = normalize_features(self.df, scaler=self.scaler)
        logger.info("MusicRecommender ready — %d songs indexed.", len(self.df))

    def recommend(
        self,
        song_name: str,
        artist_name: Optional[str] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return the top-N most similar songs to the query.

        Parameters
        ----------
        song_name   : str — title of the query song
        artist_name : str, optional — artist to disambiguate
        top_n       : int — how many recommendations to return

        Returns
        -------
        pd.DataFrame with columns:
            rank, song_name, artist_name, genre, similarity (%), distance
        """
        # ── 1. Find query song in dataset ──────────────────────────────────────
        idx = find_song_index(self.df, song_name, artist_name)
        query_vec = self.X_scaled[idx].reshape(1, -1)

        query_song   = self.df.iloc[idx]["song_name"]
        query_artist = self.df.iloc[idx].get("artist_name", "Unknown")
        logger.info("Finding songs similar to: '%s' by %s", query_song, query_artist)

        # ── 2. Query KNN — fetch n_neighbors candidates ────────────────────────
        n_fetch = min(top_n + 1, len(self.df))   # +1 because query itself appears
        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=n_fetch)

        distances = distances[0]
        indices   = indices[0]

        # ── 3. Build results — exclude the query song itself ───────────────────
        rows = []
        rank = 1
        for dist, i in zip(distances, indices):
            if int(i) == idx:
                continue    # skip the query song
            if rank > top_n:
                break

            row = self.df.iloc[i]
            # Convert cosine distance → similarity percentage
            # cosine distance ∈ [0, 2]; 0 = identical, 2 = opposite
            similarity_pct = round((1 - dist) * 100, 1)

            rows.append({
                "rank"        : rank,
                "song_name"   : row.get("song_name",   "Unknown"),
                "artist_name" : row.get("artist_name", "Unknown"),
                "genre"       : row.get("genre",       "Unknown"),
                "similarity"  : similarity_pct,
                "distance"    : round(float(dist), 6),
            })
            rank += 1

        result_df = pd.DataFrame(rows)
        return result_df

    def get_song_profile(self, song_name: str, artist_name: Optional[str] = None) -> pd.Series:
        """Return the audio feature profile for a single song."""
        idx = find_song_index(self.df, song_name, artist_name)
        return self.df.iloc[idx][AUDIO_FEATURES]


def _format_table(query: str, results: pd.DataFrame) -> str:
    """Render a clean terminal table of recommendations."""
    lines = [
        "",
        "═" * 70,
        f"  🎵  Top {len(results)} recommendations for: {query}",
        "─" * 70,
        f"  {'#':<4} {'Song':<35} {'Artist':<20} {'Similarity':>10}",
        "─" * 70,
    ]
    for _, row in results.iterrows():
        song   = str(row["song_name"])[:33]
        artist = str(row["artist_name"])[:18]
        lines.append(
            f"  {int(row['rank']):<4} {song:<35} {artist:<20} {row['similarity']:>8.1f}%"
        )
    lines.append("═" * 70)
    lines.append("")
    return "\n".join(lines)


def _parse_args():
    p = argparse.ArgumentParser(description="Get music recommendations.")
    p.add_argument("--model-dir",  default="models")
    p.add_argument("--song",       required=True,  help="Song title")
    p.add_argument("--artist",     default=None,   help="Artist name (optional)")
    p.add_argument("--top-n",      type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rec  = MusicRecommender(model_dir=args.model_dir)
    results = rec.recommend(args.song, artist_name=args.artist, top_n=args.top_n)
    label = f"{args.song}" + (f" by {args.artist}" if args.artist else "")
    print(_format_table(label, results))
