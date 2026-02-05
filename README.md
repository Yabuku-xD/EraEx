# ERAEX - The Golden Era (2012-2018)

Mood-based music discovery for SoundCloud tracks from the golden era of SoundCloud (2012-2018). Search by mood, vibe, or artist name to discover hidden gems from 14+ million curated tracks.

## Features

- **Semantic Search**: Find tracks by mood/vibe using ColBERT embeddings
- **Advanced Ranking**: Hybrid pipeline with cross-encoder re-ranking and popularity weighting
- **Dead Link Healing**: Automatically finds working alternatives for removed tracks
- **Feeling Lucky**: Personalized recommendations based on your search history
- **Year Filtering**: Filter results by release year (2012-2018)
- **SoundCloud Widget**: Inline playback using SoundCloud's embed player

## Advanced Ranking Pipeline

The search engine uses a sophisticated **6-stage hybrid pipeline**:

```
Query → Intent Classifier → Query Expansion
               ↓
     ┌─────────┴─────────┐
     ↓                   ↓
   FAISS              BM25
  (Dense)           (Sparse)
     ↓                   ↓
     └─────────┬─────────┘
               ↓
    Reciprocal Rank Fusion (RRF)
               ↓
    Cross-Encoder Re-ranking
               ↓
    MMR Diversity Filter
               ↓
        Final Results
```

1. **Intent Classification**: Detects query type (mood/artist/genre) using rule-based classifier with pre-computed mood templates.
2. **Query Expansion**: NRC Emotion Lexicon maps detected emotions to music descriptors (e.g., "i miss my ex" → "heartbreak emotional sad r&b").
3. **Hybrid Retrieval**: FAISS (dense vectors) + BM25 (sparse keywords) run in parallel.
4. **Reciprocal Rank Fusion**: Merges dense and sparse results using RRF scoring.
5. **Cross-Encoder Re-ranking**: Top 30 candidates re-scored using `ms-marco-TinyBERT-L-2-v2` (optimized for CPU).
6. **MMR Diversity**: Maximal Marginal Relevance ensures varied genres/artists in final results.

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

## Data Pipeline (via Colab)

| Step | Notebook | Description |
|------|----------|-------------|
| 0 | `00_ingest_csv.ipynb` | CSV → Parquet |
| 1 | `01_ingest_ndjson.ipynb` | NDJSON → Parquet |
| 2 | `02_filter_and_prep.ipynb` | Filter + dedupe + doc_text |
| 3 | `03_embed_colbert.ipynb` | Dense embeddings (GPU) |
| 4 | `04_build_indexes.ipynb` | FAISS + BM25 indexes |

## Stats

| Metric | Value |
|--------|-------|
| Total Tracks | 14,033,412 |
| Years | 2012-2018 |
| Embedding Model | colbert-ir/colbertv2.0 |
| Embedding Dim | 128 |
| Index Type | FAISS IVF+PQ + BM25 Hybrid |

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Search**: FAISS, BM25, ColBERT
- **Ranking**: Cross-Encoder, MMR, Reciprocal Rank Fusion
- **Data**: Polars, PyArrow
- **Link Check**: httpx, SoundCloud API v2
- **Frontend**: Jinja2, Vanilla JS/CSS