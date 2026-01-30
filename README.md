# ERAEX - The Golden Era (2013-2018)

Mood-based music discovery for SoundCloud tracks from the golden era of SoundCloud (2013-2018). Search by mood, vibe, or artist name to discover hidden gems from 14+ million curated tracks.

## Features

- **Semantic Search**: Find tracks by mood/vibe using SBERT embeddings
- **Advanced Ranking**: Hybrid pipeline with cross-encoder re-ranking and popularity weighting
- **Dead Link Healing**: Automatically finds working alternatives for removed tracks
- **Feeling Lucky**: Personalized recommendations based on your search history
- **Year Filtering**: Filter results by release year (2013-2018)
- **SoundCloud Widget**: Inline playback using SoundCloud's embed player

## Advanced Ranking Pipeline

The search engine uses a sophisticated 4-stage pipeline:

1. **Retrieval**: FAISS (IVF+PQ) finds top 200 candidates using SBERT bi-encoder.
2. **Tag Expansion**: Expands query with co-occurring tags (e.g., "chill" → "lo-fi, ambient") using a pre-computed matrix.
3. **Scoring**: Blends semantic similarity with popularity (log-normalized playback count).
4. **Re-ranking**: Top candidates are re-scored using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for high-precision ordering.

## Quick Start

### Prerequisites

- Python 3.10+
- 16GB+ RAM (for loading indexes)
- GPU recommended for embedding generation

### Installation

```bash
pip install -r requirements.txt
```

### Running the API

```bash
python run_api.py
```

Open http://localhost:8000 in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/search` | POST | Search tracks by mood/vibe |
| `/lucky` | POST | Personalized picks from history |
| `/check_links` | POST | Verify/heal SoundCloud URLs |
| `/api/search` | GET | Search via query params |

### Search Request

```json
{
  "query": "chill bedroom pop vibes",
  "year_start": 2015,
  "year_end": 2017,
  "top_k": 20
}
```

### Lucky Request

```json
{
  "history": ["late night vibes", "sad r&b", "partynextdoor"]
}
```

## Data Pipeline

| Step | Command | Description |
|------|---------|-------------|
| 1. Ingest | `python run_ingest.py` | CSV → Parquet by year |
| 2. Filter | `python run_filter.py` | Remove DJ mixes, podcasts |
| 3. Prep | `python run_prep.py` | Dedup + build doc_text |
| 4. Embed | `python run_embed.py` | SBERT vectors (GPU/Colab) |
| 5. Index | `python run_index.py` | Build FAISS IVF+PQ |

## Stats

| Metric | Value |
|--------|-------|
| Total Tracks | 14,033,412 |
| Years | 2012-2018 |
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Dim | 384 |
| Index Type | IVF+PQ |

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Search**: FAISS, Sentence-Transformers
- **Data**: Polars, PyArrow
- **Link Check**: httpx, SoundCloud API v2
- **Frontend**: Jinja2, Vanilla JS/CSS