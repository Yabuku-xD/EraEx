import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEXES_DIR = DATA_DIR / "indexes"
MUSIC_READY_DIR = PROCESSED_DIR / "music_ready"

YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018]

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 50
FAISS_NPROBE = 32

WEIGHTS = {
    "mood-heavy": {"text": 0.15, "mood": 0.60, "popularity": 0.25},
    "text-heavy": {"text": 0.50, "mood": 0.30, "popularity": 0.20},
    "genre": {"text": 0.35, "mood": 0.40, "popularity": 0.25},
    "default": {"text": 0.30, "mood": 0.45, "popularity": 0.25},
}

MMR_LAMBDA = 0.7
MIN_ARTIST_GAP = 3

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))