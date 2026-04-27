"""
tests/test_recommender.py
--------------------------
Unit tests for the Music Recommender pipeline.

Run with:
    pytest tests/ -v
"""

import math
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.preprocess import (
    normalize_features,
    find_song_index,
    AUDIO_FEATURES,
)
from src.train import train_model, save_artifacts, load_artifacts
from src.recommend import MusicRecommender


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(n: int = 30) -> pd.DataFrame:
    """Generate a synthetic songs DataFrame for testing."""
    rng = np.random.default_rng(42)
    data = {f: rng.uniform(0, 1, n) for f in AUDIO_FEATURES}
    data["loudness"] = rng.uniform(-40, 0, n)   # loudness is negative-scale
    data["song_name"]   = [f"Song {i}" for i in range(n)]
    data["artist_name"] = [f"Artist {i % 5}" for i in range(n)]
    data["genre"]       = [f"Genre {i % 3}" for i in range(n)]
    return pd.DataFrame(data)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df():
    return _make_df(30)


@pytest.fixture()
def scaled_data(sample_df):
    X_scaled, scaler = normalize_features(sample_df)
    return X_scaled, scaler


@pytest.fixture()
def trained_knn(scaled_data):
    X_scaled, _ = scaled_data
    return train_model(X_scaled)


@pytest.fixture()
def saved_dir(trained_knn, scaled_data, sample_df, tmp_path):
    X_scaled, scaler = scaled_data
    save_artifacts(trained_knn, scaler, sample_df, str(tmp_path))
    return str(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# preprocess tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeFeatures:
    def test_output_shape(self, sample_df):
        X, scaler = normalize_features(sample_df)
        assert X.shape == (len(sample_df), len(AUDIO_FEATURES))

    def test_returns_scaler(self, sample_df):
        _, scaler = normalize_features(sample_df)
        assert isinstance(scaler, StandardScaler)

    def test_zero_mean_approx(self, sample_df):
        X, _ = normalize_features(sample_df)
        assert np.abs(X.mean(axis=0)).max() < 0.1

    def test_transform_only_mode(self, sample_df):
        _, scaler = normalize_features(sample_df)
        X2, _ = normalize_features(sample_df, scaler=scaler)
        assert X2.shape == (len(sample_df), len(AUDIO_FEATURES))

    def test_missing_feature_raises(self):
        df_bad = pd.DataFrame({"song_name": ["x"], "danceability": [0.5]})
        with pytest.raises(KeyError):
            normalize_features(df_bad)


class TestFindSongIndex:
    def test_exact_match(self, sample_df):
        idx = find_song_index(sample_df, "Song 0")
        assert idx == 0

    def test_case_insensitive(self, sample_df):
        idx = find_song_index(sample_df, "song 0")
        assert idx == 0

    def test_with_artist(self, sample_df):
        idx = find_song_index(sample_df, "Song 0", "Artist 0")
        assert idx == 0

    def test_not_found_raises(self, sample_df):
        with pytest.raises(ValueError):
            find_song_index(sample_df, "Nonexistent Song XYZ")


# ══════════════════════════════════════════════════════════════════════════════
# train tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainModel:
    def test_returns_nearest_neighbors(self, scaled_data):
        X, _ = scaled_data
        knn  = train_model(X)
        assert isinstance(knn, NearestNeighbors)

    def test_kneighbors_returns_correct_count(self, scaled_data):
        X, _    = scaled_data
        knn     = train_model(X)
        dists, idxs = knn.kneighbors(X[:1], n_neighbors=5)
        assert dists.shape == (1, 5)
        assert idxs.shape  == (1, 5)

    def test_self_distance_is_zero(self, scaled_data):
        X, _  = scaled_data
        knn   = train_model(X)
        dists, _ = knn.kneighbors(X[:1], n_neighbors=1)
        assert dists[0, 0] == pytest.approx(0.0, abs=1e-6)


class TestArtifacts:
    def test_files_exist(self, saved_dir):
        assert os.path.exists(os.path.join(saved_dir, "knn_model.joblib"))
        assert os.path.exists(os.path.join(saved_dir, "scaler.joblib"))
        assert os.path.exists(os.path.join(saved_dir, "songs_df.joblib"))

    def test_load_types(self, saved_dir):
        knn, scaler, df = load_artifacts(saved_dir)
        assert isinstance(knn,    NearestNeighbors)
        assert isinstance(scaler, StandardScaler)
        assert isinstance(df,     pd.DataFrame)

    def test_missing_dir_raises(self, tmp_path):
        empty = str(tmp_path / "empty")
        with pytest.raises(FileNotFoundError):
            load_artifacts(empty)


# ══════════════════════════════════════════════════════════════════════════════
# recommend tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMusicRecommender:
    @pytest.fixture()
    def recommender(self, saved_dir):
        return MusicRecommender(model_dir=saved_dir)

    def test_returns_dataframe(self, recommender):
        result = recommender.recommend("Song 0", top_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_results(self, recommender):
        result = recommender.recommend("Song 0", top_n=5)
        assert len(result) == 5

    def test_result_columns_present(self, recommender):
        result = recommender.recommend("Song 0", top_n=3)
        for col in ["rank", "song_name", "artist_name", "similarity", "distance"]:
            assert col in result.columns

    def test_similarity_in_valid_range(self, recommender):
        result = recommender.recommend("Song 0", top_n=5)
        assert result["similarity"].between(0, 100).all()

    def test_query_song_not_in_results(self, recommender):
        result = recommender.recommend("Song 0", top_n=10)
        assert "Song 0" not in result["song_name"].values

    def test_results_sorted_by_similarity(self, recommender):
        result = recommender.recommend("Song 0", top_n=8)
        sims   = result["similarity"].tolist()
        assert sims == sorted(sims, reverse=True)

    def test_invalid_song_raises(self, recommender):
        with pytest.raises(ValueError):
            recommender.recommend("ZZZ_Nonexistent_Song_9999")

    def test_song_profile_returns_series(self, recommender):
        profile = recommender.get_song_profile("Song 1")
        assert isinstance(profile, pd.Series)
        assert set(AUDIO_FEATURES).issubset(set(profile.index))
