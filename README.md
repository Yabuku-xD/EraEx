# EraEx

EraEx is a full-stack music discovery app that combines semantic search,
adaptive recommendations, media enrichment, and a multi-view Flask UI.

## Table of Contents

- [What It Includes](#what-it-includes)
- [Architecture Snapshot](#architecture-snapshot)
- [Screenshots](#screenshots)
- [Install](#install)
- [Run](#run)
- [Project Map](#project-map)
- [Workflows](#workflows)
- [Runtime Data](#runtime-data)
- [API Highlights](#api-highlights)
- [Validation Snapshot](#validation-snapshot)
- [Contributing](#contributing)

## What It Includes

- Semantic retrieval powered by `BAAI/bge-m3` embeddings and FAISS indexes.
- Personalized "For You" ranking with cold-start and behavior-aware flows.
- Media enrichment and playback fallback through `yt-dlp`, YouTube metadata,
  optional Spotify cover/playback integration, and lyrics lookups.
- A server-rendered web UI with Search, Home, For You, Liked, History, and
  Playlists views.
- CLI and notebook workflows for maintenance, validation, tuning, and index
  rebuilding.

## Architecture Snapshot

```text
+-------------------+      +------------------------+
| Flask UI + API    | ---> | search_pipeline.py     |
| web_app.py        |      | semantic + hybrid rank |
+---------+---------+      +-----------+------------+
          |                            |
          |                            v
          |                 +------------------------+
          |                 | FAISS + metadata       |
          |                 | data/indexes/*         |
          |                 +------------------------+
          |
          +---------------> +------------------------+
          |                 | recommendation_engine  |
          |                 | cold start + adaptive  |
          |                 +-----------+------------+
          |                             |
          v                             v
+-------------------+        +-----------------------+
| SQLite profiles   |        | Media enrichment      |
| likes/plays/auth  |        | yt-dlp / Spotify /    |
| data/users.db     |        | lyrics providers      |
+-------------------+        +-----------------------+
```

Deeper docs:

- [Architecture](/Users/Yabuku/Downloads/EraEx/docs/architecture.md)
- [Repository Map](/Users/Yabuku/Downloads/EraEx/docs/repo-map.md)

## Screenshots

### Home
![EraEx Home](/Users/Yabuku/Downloads/EraEx/assets/home.png)

### Search
![EraEx Search](/Users/Yabuku/Downloads/EraEx/assets/search.png)

### For You
![EraEx For You](/Users/Yabuku/Downloads/EraEx/assets/foryou.png)

## Install

### Dependencies

- Python 3.10+
- `yt-dlp` available on `PATH` for media resolution and metadata enrichment
- Runtime index artifacts under `data/indexes/`

### Virtual Environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Git Bash, WSL, Linux, or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```bash
python run.py
```

The server warms the search and recommendation stacks on startup and prints the
local address to open in your browser.

## Project Map

Top-level layout:

```text
EraEx/
  assets/       screenshots for repository docs
  cli_tools/    debug, validation, tuning, and optimization scripts
  config/       runtime settings and environment parsing
  data/         indexes, caches, eval artifacts, and user SQLite data
  notebooks/    rebuild and analysis notebooks
  src/          product code
  run.py        app warmup + Flask entrypoint
```

Code ownership by area:

- `src/web_api/`: Flask routes, orchestration, auth/session handling.
- `src/search/`: semantic retrieval, candidate union, reranking, diagnostics.
- `src/recommendation/`: cold-start and adaptive recommendation logic.
- `src/core/`: embeddings, media metadata, lyrics, lazy loading, audio logic.
- `src/user_profiles/`: SQLite-backed likes, plays, skips, playlists, accounts.
- `src/templates/` and `src/static/`: server-rendered frontend shell and styles.

## Workflows

Search debugging:

```bash
python -m cli_tools.search_query_cli "midnight drive rnb trap" --limit 10
python -m cli_tools.search_query_cli --artist "Miguel" --song "coffee"
```

Recommendation debugging:

```bash
python -m cli_tools.foryou_recommendation_cli --user-id <USER_ID>
python -m cli_tools.foryou_recommendation_cli --username <USERNAME> --fast
```

Maintenance and evaluation:

```bash
python cli_tools/project_maintenance_cli.py enrich --only-missing --limit 1000
python cli_tools/project_maintenance_cli.py audio-features
python cli_tools/project_maintenance_cli.py simulate
python cli_tools/bulk_validation_cli.py
```

Index and notebook workflows:

- `notebooks/full_pipeline.ipynb`: rebuild metadata, embeddings, and FAISS.
- `notebooks/build_ultrafast_ivf_index.py`: build the optimized IVF index path.
- `notebooks/reembed_reindex_colab.ipynb`: remote or Colab rebuild support.

## Runtime Data

EraEx expects non-trivial runtime assets that are usually kept out of Git:

- `data/indexes/metadata.json`
- `data/indexes/faiss_index.bin` or `data/indexes/faiss_index_ivf.bin`
- `data/indexes/id_map.json` or `data/indexes/track_ids.pkl`
- `data/cache/*` for recommendation and feature caches

Repository-tracked data:

- `data/eval/`: validation plans and answer sets
- `data/users.db`: local SQLite profile store used by the app

## API Highlights

- `GET /search`: hybrid search tuned for explicit queries.
- `GET /sonic`: semantic discovery flow used by the Home experience.
- `GET /api/recommend`: adaptive or cold-start recommendations.
- `GET /api/trending`: popularity-based fallback feed.
- `GET /api/resolve_video`: media playback fallback resolution.
- `GET /api/track_enrich`: on-demand metadata enrichment.
- `GET /api/lyrics`: lyrics lookup with cache and provider fallback.
- `POST /api/like`, `POST /api/play`, `POST /api/skip`: user feedback capture.
- `GET /api/playlists` and `/api/auth/*`: library and account flows.

## Validation Snapshot

Current tracked validation evidence:

- `metrics.md` summarizes search, recommendation, and bot validation results.
- `data/eval/bulk_validation_answers.json` stores the larger bulk validation run.

Highlighted metrics from the current repo snapshot:

- Home semantic queries: `100%` success rate across 90 queries.
- Search query evaluation: `40` artist/song-style queries in the bulk report.
- For You validation: `24` synthetic users plus a real-profile evaluation block.

## Contributing

The repo is currently optimized for direct engineering iteration rather than a
formal contributor workflow. If you extend it, keep docs and validation assets
aligned with code changes, and avoid committing large local indexes or machine-
specific environment files.
