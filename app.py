"""
app.py
------
Interactive Music Recommender Web Application built with Streamlit.

Run with:
    streamlit run app.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src modules are discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.explain import explain_recommendation
from src.preprocess import AUDIO_FEATURES, find_song_index, prepare_data
from src.recommend import MusicRecommender
from src.train import load_artifacts, save_artifacts, train_model
from src.visualize import plot_feature_radar, plot_recommendations

st.set_page_config(
    page_title="Music Recommender Engine",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Music Recommender Engine")
st.markdown(
    "> **Content-based recommendation engine** built with **K-Nearest Neighbors** and Spotify audio features."
)

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "results"


@st.cache_resource
def get_pipeline():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if artifacts exist, else train on data
    knn_path = os.path.join(MODEL_DIR, "knn_model.joblib")
    if not os.path.exists(knn_path):
        data = prepare_data(DATA_DIR)
        knn = train_model(data["X_scaled"])
        save_artifacts(knn, data["scaler"], data["df"], MODEL_DIR)

    knn, scaler, df = load_artifacts(MODEL_DIR)
    recommender = MusicRecommender(MODEL_DIR)
    return recommender, df


try:
    recommender, df = get_pipeline()
except Exception as e:
    st.error(f"Error initializing recommender: {e}")
    st.stop()

# Sidebar: Selection
st.sidebar.header("🎧 Song Selection")
song_options = [
    f"{row['song_name']} — {row['artist_name']}" for _, row in df.iterrows()
]
selected_option = st.sidebar.selectbox("Choose a track to analyze:", song_options)
top_k = st.sidebar.slider("Number of recommendations:", 3, 10, 5)

if selected_option:
    selected_idx = song_options.index(selected_option)
    selected_row = df.iloc[selected_idx]
    song_name = selected_row["song_name"]
    artist_name = selected_row["artist_name"]

    st.subheader(f"Analyzing: **{song_name}** by *{artist_name}*")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Genre", str(selected_row.get("genre", "Unknown")))
    with col2:
        st.metric("Energy", f"{selected_row.get('energy', 0):.2f}")
    with col3:
        st.metric("Danceability", f"{selected_row.get('danceability', 0):.2f}")
    with col4:
        st.metric("Valence (Mood)", f"{selected_row.get('valence', 0):.2f}")

    # Recommendations
    recs = recommender.recommend_by_index(selected_idx, n_recommendations=top_k)

    st.markdown("### 🌟 Recommended Similar Tracks")
    rec_cols = st.columns(min(top_k, 5))
    for i, rec in enumerate(recs[:5]):
        with rec_cols[i]:
            st.info(
                f"**#{i+1} {rec['song_name']}**\n\n"
                f"👤 *{rec['artist_name']}*\n\n"
                f"🏷️ `{rec.get('genre', 'Music')}`\n\n"
                f"🎯 **{rec['similarity_pct']}% Match**"
            )

    # Detailed Table & Radar Plot
    st.markdown("---")
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("#### 📋 Detailed Recommendations Table")
        rec_df = pd.DataFrame(recs)[
            ["rank", "song_name", "artist_name", "genre", "similarity_pct"]
        ]
        rec_df.columns = ["Rank", "Song Title", "Artist", "Genre", "Similarity %"]
        st.dataframe(rec_df, use_container_width=True)

    with right_col:
        st.markdown("#### 📡 Feature Comparison Radar")
        if recs:
            top_rec = recs[0]
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

            categories = ["Dance", "Energy", "Acoustic", "Valence", "Speech"]
            # Normalize display features between 0 and 1
            query_vals = [
                float(selected_row.get("danceability", 0.5)),
                float(selected_row.get("energy", 0.5)),
                float(selected_row.get("acousticness", 0.5)),
                float(selected_row.get("valence", 0.5)),
                float(selected_row.get("speechiness", 0.5)),
            ]
            rec_row = df.iloc[top_rec["song_index"]]
            rec_vals = [
                float(rec_row.get("danceability", 0.5)),
                float(rec_row.get("energy", 0.5)),
                float(rec_row.get("acousticness", 0.5)),
                float(rec_row.get("valence", 0.5)),
                float(rec_row.get("speechiness", 0.5)),
            ]

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            query_vals += query_vals[:1]
            rec_vals += rec_vals[:1]
            angles += angles[:1]

            ax.plot(angles, query_vals, color="#1DB954", linewidth=2, label="Query")
            ax.fill(angles, query_vals, color="#1DB954", alpha=0.25)
            ax.plot(angles, rec_vals, color="#4C72B0", linewidth=2, label="Top Match")
            ax.fill(angles, rec_vals, color="#4C72B0", alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
            st.pyplot(fig)
            plt.close(fig)
