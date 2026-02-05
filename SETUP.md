# EraEx Setup Guide

## Prerequisites
- Python 3.10+
- Google Colab (GPU for embedding)
- 16GB+ RAM locally
- **Internet connection** (for first run to download NLTK data)

---

## Step 1: Upload to Google Drive

Upload `EraEx` folder to:
```
MyDrive/EraEx/
├── data/raw/
│   ├── dataset.csv           (22GB - existing)
│   └── 1-100m.ndjson.zst     (10.7GB)
└── notebooks/
```

---

## Step 2: Run Colab Notebooks (In Order)

| # | Notebook | Purpose | Runtime |
|---|----------|---------|---------|
| 1 | `01_ingest_ndjson.ipynb` | NDJSON → CSV | 1 hr / CPU |
| 2 | `00_ingest_csv.ipynb` | Both CSVs → Parquet | 40 min / CPU |
| 3 | `02_filter_and_prep.ipynb` | Filter + dedupe | 30 min / CPU |
| 4 | `03_embed_colbert.ipynb` | ColBERT embeddings | 3 hrs / **GPU** |
| 5 | `04_build_indexes.ipynb` | FAISS + BM25 | 30 min / CPU |

---

## Step 3: Download Results

After notebooks complete, download from Drive:
```
EraEx/data/
├── embeddings/          ← Download
├── indexes/             ← Download
└── processed/music_ready/  ← Download
```

---

## Step 4: Run Locally

1. Install dependencies:
```bash
cd EraEx
pip install -r requirements.txt
python -m textblob.download_corpora  # Optional: for extra NLTK data
```

2. Run the API:
```bash
python run_api.py
```

3. Open **http://localhost:8000**

---

## Architecture & Automation

- **Fully Automated Search**: No manual keyword lists.
- **NRCLex**: Detects emotions from 14,000+ words.
- **WordNet**: Expands synonyms automatically.
- **ColBERT**: Semantic embedding for search.

## Expected Latency

| Component | Time |
|-----------|------|
| ColBERT embedding | 50ms |
| FAISS + BM25 search | 110ms |
| RRF fusion | 5ms |
| TinyBERT reranker | 150ms |
| MMR diversity | 20ms |
| **Total** | **~340ms** |
