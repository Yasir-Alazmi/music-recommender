"""
visualize.py
------------
Visualization module:
  1. PCA 2D Scatter — projects all songs into 2D space, coloured by genre.
  2. Recommendation Bar Chart — shows similarity scores for top-N results.
  3. Feature Radar Chart — compares query song vs recommended songs on each feature.

Usage:
    python -m src.visualize --model-dir models/ --song "Blinding Lights" --output-dir results/
"""

import argparse
import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.train import load_artifacts
from src.preprocess import normalize_features, AUDIO_FEATURES, find_song_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Palette ────────────────────────────────────────────────────────────────────
DARK_BG    = "#0f0f1a"
PANEL_BG   = "#1a1a2e"
ACCENT     = "#e94560"
TEXT_COLOR = "#e0e0f0"
GRID_COLOR = "#2a2a4a"

GENRE_PALETTE = [
    "#e94560", "#0f3460", "#533483", "#e94560",
    "#2ecc71", "#f39c12", "#3498db", "#9b59b6",
    "#1abc9c", "#e74c3c", "#34495e", "#f1c40f",
]


def _style_axis(ax):
    """Apply dark theme to a matplotlib axis."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)


def plot_pca_clusters(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    query_idx: Optional[int] = None,
    recommended_indices: Optional[List[int]] = None,
    output_dir: str = "results",
    filename: str = "pca_clusters.png",
) -> str:
    """
    Project the full song dataset into 2D with PCA and plot coloured by genre.

    PCA recap
    ---------
    • Principal Component Analysis finds the axes of greatest variance.
    • We compress 9 audio features → 2 dimensions for visualization.
    • We lose some information but gain interpretability.

    Parameters
    ----------
    df                   : songs DataFrame
    X_scaled             : normalized feature matrix
    query_idx            : index of the query song (highlighted in plot)
    recommended_indices  : indices of recommended songs (highlighted)
    output_dir           : where to save the PNG
    filename             : output filename

    Returns
    -------
    Path to the saved PNG.
    """
    logger.info("Running PCA (9D → 2D) for visualization …")
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var    = pca.explained_variance_ratio_

    genres      = df["genre"].fillna("Unknown").astype(str).tolist()
    unique_genres = sorted(set(genres))
    genre_to_idx  = {g: i for i, g in enumerate(unique_genres)}
    color_indices = [genre_to_idx[g] for g in genres]
    colors        = [GENRE_PALETTE[i % len(GENRE_PALETTE)] for i in color_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(DARK_BG)
    _style_axis(ax)

    # ── All songs ──────────────────────────────────────────────────────────────
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, alpha=0.35, s=8, linewidths=0,
    )

    # ── Recommended songs ──────────────────────────────────────────────────────
    if recommended_indices:
        rx = coords[recommended_indices, 0]
        ry = coords[recommended_indices, 1]
        ax.scatter(rx, ry, c="#f39c12", s=60, zorder=4,
                   edgecolors="white", linewidths=0.5, label="Recommended")

    # ── Query song ─────────────────────────────────────────────────────────────
    if query_idx is not None:
        ax.scatter(
            coords[query_idx, 0], coords[query_idx, 1],
            c=ACCENT, s=180, zorder=5, marker="*",
            edgecolors="white", linewidths=0.8, label="Query Song",
        )
        ax.annotate(
            df.iloc[query_idx]["song_name"],
            (coords[query_idx, 0], coords[query_idx, 1]),
            textcoords="offset points", xytext=(8, 8),
            color="white", fontsize=8, fontweight="bold",
        )

    # ── Legend patches per genre ───────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=GENRE_PALETTE[i % len(GENRE_PALETTE)], label=g)
        for i, g in enumerate(unique_genres[:12])
    ]
    ax.legend(
        handles=patches, loc="upper right", fontsize=7,
        facecolor=PANEL_BG, edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR, ncol=2,
    )

    ax.set_title(
        f"Song Clusters — PCA Projection  "
        f"(PC1: {var[0]*100:.1f}%  PC2: {var[1]*100:.1f}% variance explained)",
        fontsize=12, pad=14,
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info("PCA plot saved → %s", out)
    return out


def plot_recommendations(
    query_song: str,
    results_df: pd.DataFrame,
    output_dir: str = "results",
    filename: str = "recommendations.png",
) -> str:
    """
    Horizontal bar chart showing similarity scores of top-N recommendations.
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(results_df) * 0.55)))
    fig.patch.set_facecolor(DARK_BG)
    _style_axis(ax)

    labels = [
        f"{row['song_name'][:30]}  —  {row['artist_name'][:18]}"
        for _, row in results_df.iterrows()
    ]
    scores = results_df["similarity"].tolist()

    # Gradient colours from low to high similarity
    norm_scores = [(s - min(scores)) / (max(scores) - min(scores) + 1e-9)
                   for s in scores]
    bar_colors  = plt.cm.plasma(norm_scores)

    bars = ax.barh(labels[::-1], scores[::-1], color=bar_colors[::-1],
                   edgecolor="none", height=0.65)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%", va="center", ha="left",
            color=TEXT_COLOR, fontsize=9,
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Similarity Score (%)")
    ax.set_title(f"Top {len(results_df)} Recommendations for: {query_song}",
                 fontsize=12, pad=12)
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", linewidth=0.5)
    ax.set_facecolor(PANEL_BG)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info("Recommendation chart saved → %s", out)
    return out


def plot_feature_radar(
    query_profile: pd.Series,
    recommended_profiles: List[pd.Series],
    rec_labels: List[str],
    query_label: str,
    output_dir: str = "results",
    filename: str = "feature_radar.png",
) -> str:
    """
    Radar (spider) chart comparing audio features of query vs top recommendations.
    """
    features = [f for f in AUDIO_FEATURES if f != "loudness"]   # loudness is negative-scale
    N        = len(features)
    angles   = [n / float(N) * 2 * np.pi for n in range(N)]
    angles  += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5)

    # ── Query song ─────────────────────────────────────────────────────────────
    q_vals  = [float(query_profile.get(f, 0)) for f in features]
    q_vals += q_vals[:1]
    ax.plot(angles, q_vals, color=ACCENT, linewidth=2.5, label=query_label)
    ax.fill(angles, q_vals, color=ACCENT, alpha=0.25)

    # ── Top 3 recommendations ──────────────────────────────────────────────────
    colors = ["#3498db", "#2ecc71", "#f39c12"]
    for i, (profile, label) in enumerate(zip(recommended_profiles[:3], rec_labels[:3])):
        vals  = [float(profile.get(f, 0)) for f in features]
        vals += vals[:1]
        c = colors[i % len(colors)]
        ax.plot(angles, vals, color=c, linewidth=1.5, linestyle="--", label=label[:25])
        ax.fill(angles, vals, color=c, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, color=TEXT_COLOR, fontsize=9)
    ax.set_title("Feature Radar: Query vs Recommendations",
                 color=TEXT_COLOR, fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info("Radar chart saved → %s", out)
    return out


def _parse_args():
    p = argparse.ArgumentParser(description="Generate visualizations for the music recommender.")
    p.add_argument("--model-dir",  default="models")
    p.add_argument("--song",       default=None, help="Query song for highlights")
    p.add_argument("--artist",     default=None)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    knn, scaler, df = load_artifacts(args.model_dir)
    X_scaled, _     = normalize_features(df, scaler=scaler)

    query_idx = None
    if args.song:
        try:
            query_idx = find_song_index(df, args.song, args.artist)
        except ValueError as e:
            logger.warning(str(e))

    plot_pca_clusters(df, X_scaled, query_idx=query_idx, output_dir=args.output_dir)
    logger.info("Visualizations complete.")
