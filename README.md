# EraEx

EraEx is a Flask-based music discovery app with semantic search, adaptive recommendations, profile-driven playlists, and a web player with YouTube/Spotify fallback support.

## Screenshots

### Home
![EraEx Home](assets/home.png)

### Search
![EraEx Search](assets/search.png)

### For You
![EraEx For You](assets/foryou.png)

### Liked
![EraEx Liked](assets/liked.png)

### History
![EraEx History](assets/history.png)

### Playlists
![EraEx Playlists](assets/playlists.png)

## What Is In The Codebase

- Flask web app with server-rendered templates and modular static JS/CSS.
- Semantic retrieval pipeline in `src/search/search_pipeline.py` using `BAAI/bge-m3` + FAISS.
- Recommendation engine in `src/recommendation/recommendation_engine.py` with cold-start and adaptive profile modes.
- User/account/profile storage in SQLite via `src/user_profiles/user_profile_store.py`.
- Media enrichment/resolution in `src/core/media_metadata.py` (yt-dlp fallback, thumbnail candidates, optional Spotify cover resolution).
- Optional lyrics fetch pipeline (`/api/lyrics`) with caching and Lyrica/LrcLib fallbacks.

## Prerequisites

- Python 3.10+ recommended.
- `yt-dlp` available (package is pinned in `requirements.txt`; system binary fallback is also supported).

## Install

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bash / WSL / Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Required Local Data

Search/recommendation runtime expects index artifacts under `data/indexes`:

- `faiss_index.bin` or `faiss.index`
- `id_map.json` or `track_ids.pkl`
- `metadata.json`

You can rebuild artifacts with `notebooks/full_pipeline.ipynb`.

## Run The App

```bash
python run.py
```

Open `http://127.0.0.1:5000` (or `http://localhost:5000`).

## CLI Tools

### Search Query CLI

```bash
python -m cli_tools.search_query_cli "midnight drive rnb trap" --limit 10
python -m cli_tools.search_query_cli --artist "Miguel" --song "coffee"
```

### For You Recommendation CLI

```bash
python -m cli_tools.foryou_recommendation_cli --user-id <USER_ID>
python -m cli_tools.foryou_recommendation_cli --username <USERNAME> --password <PASSWORD> --fast
```

### Search Tuning CLI

```bash
python -m cli_tools.search_tuning_cli queries.json --trials 60 --report-json tuning_report.json
python -m cli_tools.search_tuning_cli --template-out queries_template.json dummy.json
```

### Maintenance CLI

```bash
python cli_tools/project_maintenance_cli.py enrich --only-missing --limit 1000
python cli_tools/project_maintenance_cli.py audio-features
python cli_tools/project_maintenance_cli.py simulate
python cli_tools/project_maintenance_cli.py subset-test --subset-size 5
python cli_tools/project_maintenance_cli.py clean --dry-run
```

## API Surface (Main Routes)

- `GET /search`
- `GET /sonic`
- `GET /api/recommend`
- `GET /api/trending`
- `GET /api/resolve_video`
- `GET /api/track_enrich`
- `GET /api/lyrics`
- `POST /api/like`
- `POST /api/unlike`
- `POST /api/dislike`
- `POST /api/undislike`
- `POST /api/play`
- `POST /api/skip`
- `GET /api/liked`
- `GET /api/history`
- `GET /api/playlists` and playlist mutation routes under `/api/playlists/*`
- Auth routes under `/api/auth/*`
- Spotify integration routes under `/api/spotify/*` plus `/spotify/callback`

## Key Environment Variables

- `EMBEDDING_MODEL` (default `BAAI/bge-m3`)
- `FLASK_SECRET_KEY`
- `SESSION_COOKIE_SECURE`
- `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_PLAYBACK_ENABLED`, `SPOTIFY_REDIRECT_URI`
- `LYRICA_BASE_URL` (optional sidecar service)
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (for notebook/model workflows that require gated Hugging Face access)

## Project Layout

```text
EraEx/
  assets/                      # README screenshots
  cli_tools/                   # search, recommendation, tuning, maintenance CLIs
  config/settings.py           # global runtime settings
  notebooks/                   # full pipeline and analysis notebooks
  src/
    core/                      # embeddings, media metadata, lazy loading, audio features
    search/                    # semantic retrieval + ranking pipeline
    recommendation/            # recommendation engine
    user_profiles/             # SQLite-backed profile/account storage
    web_api/                   # Flask app and routes
    templates/ + static/       # frontend UI
  run.py                       # startup + warmup entrypoint
```
