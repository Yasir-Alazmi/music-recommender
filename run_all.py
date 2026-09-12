"""
run_all.py
----------
Improvement #2: One-command pipeline runner.
Chains: prepare → train → visualize → recommend → explain.

Usage:
    python run_all.py --data-dir data/ --model-dir models/ --output-dir results/ --song "Shape of You"
"""

import argparse
import logging
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.preprocess import prepare_data, find_song_index
from src.train import train_model, save_artifacts
from src.recommend import MusicRecommender, _format_table
from src.visualize import plot_pca_clusters, plot_recommendations, plot_feature_radar
from src.explain import explain_recommendation

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEMO_SONGS = [
    ("Shape of You", "Ed Sheeran"),
    ("Blinding Lights", "The Weeknd"),
    ("Bohemian Rhapsody", "Queen"),
]


def _banner(text: str) -> None:
    w = 60
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def run_pipeline(data_dir, model_dir, output_dir, demo_song=None, demo_artist=None):
    t0 = time.perf_counter()

    # ── Phase 1: Data ──────────────────────────────────────────────────────────
    _banner("Phase 1 / 4 — Loading & Normalizing Data")
    data = prepare_data(data_dir)

    # ── Phase 2: Training ──────────────────────────────────────────────────────
    _banner("Phase 2 / 4 — Training KNN Model")
    knn = train_model(data["X_scaled"])
    save_artifacts(knn, data["scaler"], data["df"], model_dir)

    # ── Phase 3: Visualizations ────────────────────────────────────────────────
    _banner("Phase 3 / 4 — Generating Visualizations")
    query_idx = None
    if demo_song:
        try:
            query_idx = find_song_index(data["df"], demo_song, demo_artist)
        except ValueError as e:
            logger.warning(str(e))

    plot_pca_clusters(
        data["df"], data["X_scaled"],
        query_idx=query_idx, output_dir=output_dir,
    )

    # ── Phase 4: Recommendations ───────────────────────────────────────────────
    _banner("Phase 4 / 4 — Demo Recommendations")
    rec = MusicRecommender(model_dir=model_dir)

    songs_to_demo = [(demo_song, demo_artist)] if demo_song else DEMO_SONGS
    for song, artist in songs_to_demo:
        try:
            results = rec.recommend(song, artist_name=artist, top_n=10)
            label   = f"{song}" + (f" by {artist}" if artist else "")
            print(_format_table(label, results))

            plot_recommendations(song, results, output_dir=output_dir)

            # Radar chart: query vs top 3
            query_profile  = rec.get_song_profile(song, artist)
            rec_profiles   = [rec.get_song_profile(r["song_name"], r["artist_name"])
                              for _, r in results.head(3).iterrows()]
            rec_labels     = [r["song_name"] for _, r in results.head(3).iterrows()]
            plot_feature_radar(query_profile, rec_profiles, rec_labels,
                               query_label=song, output_dir=output_dir)

            print(explain_recommendation(song, artist_name=artist,
                                         top_n=5, model_dir=model_dir))
        except ValueError as e:
            logger.warning("Skipping demo song: %s", e)

    elapsed = time.perf_counter() - t0
    _banner(f"Pipeline Complete — {elapsed:.1f}s total")
    print(f"  Songs indexed : {len(data['df'])}")
    print(f"  Model saved   : {model_dir}/knn_model.joblib")
    print(f"  Plots saved   : {output_dir}/")
    print("═" * 60 + "\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Run the full Music Recommender pipeline.")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--model-dir",  default="models")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--song",       default=None, help="Demo song title")
    p.add_argument("--artist",     default=None, help="Demo artist name")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.data_dir, args.model_dir, args.output_dir,
                 demo_song=args.song, demo_artist=args.artist)
