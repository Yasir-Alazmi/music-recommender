# 🎵 Music Recommender System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![CI](https://github.com/Yasir-Alazmi/music-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Alazmi/music-recommender/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-23%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A content-based music recommendation engine built with **K-Nearest Neighbors**, Spotify audio features, and rich visualizations.

> **Given any song, find the most musically similar tracks — by danceability, energy, tempo, valence, and more. Built-in sample dataset enabled for zero-setup execution.**

---

## 📁 Project Structure

```
music-recommender/
├── data/                    # Place spotify_songs.csv here
├── models/                  # Saved KNN model, scaler, song index
├── results/                 # PCA cluster plot, bar charts, radar charts
├── src/
│   ├── preprocess.py        # Data loading, cleaning, StandardScaler normalization
│   ├── train.py             # KNN model training and artifact persistence
│   ├── recommend.py         # MusicRecommender class — core recommendation engine
│   ├── visualize.py         # PCA cluster, bar chart, radar chart visualizations
│   └── explain.py           # Feature-level explainability for each recommendation
├── tests/
│   └── test_recommender.py  # 22 unit tests with pytest
├── run_all.py               # One-command pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Yasir-Alazmi/music-recommender.git
cd music-recommender
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Download Dataset
From Kaggle: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

Place it as:
```
data/spotify_songs.csv
```

---

## 🚀 Usage

### Run the full pipeline
```bash
python run_all.py --data-dir data/ --model-dir models/ --output-dir results/ --song "Shape of You" --artist "Ed Sheeran"
```

### Train only
```bash
python -m src.train --data-dir data/ --model-dir models/
```

### Get recommendations
```bash
python -m src.recommend --model-dir models/ --song "Blinding Lights" --artist "The Weeknd" --top-n 10
```

**Output:**
```
══════════════════════════════════════════════════════════════════════
  🎵  Top 10 recommendations for: Blinding Lights
──────────────────────────────────────────────────────────────────────
  #    Song                                Artist               Similarity
──────────────────────────────────────────────────────────────────────
  1    Save Your Tears                     The Weeknd              97.3%
  2    Starboy                             The Weeknd              95.1%
  ...
══════════════════════════════════════════════════════════════════════
```

### Explain a recommendation
```bash
python -m src.explain --model-dir models/ --song "Blinding Lights" --top-n 5
```

### Visualize clusters
```bash
python -m src.visualize --model-dir models/ --song "Bohemian Rhapsody" --output-dir results/
```

### Run tests
```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🧠 How It Works

| Stage | Tool | Purpose |
|---|---|---|
| Feature extraction | `pandas` | Load 9 audio features per song |
| Normalization | `StandardScaler` | Equal-weight all features for KNN distance |
| Model | `NearestNeighbors` | Cosine distance similarity lookup |
| Persistence | `joblib` | Save model, scaler, and song index |
| Visualization | `matplotlib` + `PCA` | Cluster plots and radar charts |
| Explainability | Feature-wise diff | Show which features drove similarity |

### Why Cosine Distance?
- Cosine measures the **angle** between two vectors — ignores magnitude
- Better for audio features than Euclidean: a quiet and a loud song with the same proportional feature mix should be considered similar

### Why StandardScaler?
- `loudness` ranges from −60 to 0 dB
- `danceability` ranges from 0 to 1
- Without scaling, loudness would dominate all distance calculations

---

## 📊 Visualizations Generated

| Plot | File | What it shows |
|---|---|---|
| PCA Clusters | `results/pca_clusters.png` | All songs in 2D, coloured by genre |
| Recommendations | `results/recommendations.png` | Similarity bar chart for top-N results |
| Feature Radar | `results/feature_radar.png` | Query vs top-3 audio feature comparison |

---

## 🔧 Configuration

**`preprocess.py`**
```python
AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
```

**`train.py`**
```python
KNN_N_NEIGHBORS = 20       # fetch pool size
KNN_METRIC      = "cosine" # distance metric
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| scikit-learn | KNN model, PCA, StandardScaler |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | All visualizations |
| joblib | Model serialization |
| pytest | Unit testing |

---

## 👤 Author

**Yasir Al-Azmi** · AI Student  
GitHub: [@Yasir-Alazmi](https://github.com/Yasir-Alazmi)
