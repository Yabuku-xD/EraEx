# EraEx

EraEx is a music discovery and recommendation app built around:

1. Semantic search with `BAAI/bge-m3` + FAISS.
2. DPP-based cold start and adaptive recommendations.
3. YouTube playback with automatic fallback video resolution via `yt-dlp`.

## Setup

### Option A: No Virtual Environment

Install dependencies globally if your Python setup allows it, then make sure `yt-dlp` is on `PATH`.

Windows `yt-dlp` from GitHub:

1. Download `yt-dlp.exe` from `https://github.com/yt-dlp/yt-dlp/releases/latest`
2. Place it in a folder already in `PATH`, or add its folder to `PATH`
3. Verify with:

```powershell
yt-dlp --version
```

### Option B: Virtual Environment

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Git Bash / WSL / Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### If You Are in PowerShell

Do not use `source` or `src`. Use:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run

```bash
python run.py
```

Open `http://localhost:5000`.

## Maintenance Commands

1. One-shot full rebuild (metadata + id_map + embeddings + FAISS): `notebooks/full_pipeline.ipynb`
2. Enrich metadata with yt-dlp (description + instrumental): `python cli_tools/project_maintenance_cli.py enrich --only-missing --limit 1000`
3. Extract audio feature cache: `python cli_tools/project_maintenance_cli.py audio-features`
4. Run recommendation simulation: `python cli_tools/project_maintenance_cli.py simulate`
5. Run subset pipeline smoke test: `python cli_tools/project_maintenance_cli.py subset-test --subset-size 5`
6. Remove temp/unused local artifacts: `python cli_tools/project_maintenance_cli.py clean`

## Key API Endpoints

1. `GET /search?q=...&limit=...`
2. `GET /sonic?q=...&limit=...`
3. `GET /api/recommend?user_id=...&n=10`
4. `GET /api/trending?n=10`
5. `GET /api/resolve_video?title=...&artist=...&track_id=...`
6. `GET /api/track_enrich?track_id=...&title=...&artist=...`
7. `POST /api/like`
8. `POST /api/play`
9. `POST /api/unlike`

## Notes

1. `requirements.txt` pins `yt-dlp` directly from GitHub.
2. If metadata has empty `cover_url`, the app falls back to YouTube thumbnails.
3. If a returned `video_id` fails in the player, frontend retries using `/api/resolve_video`.
4. Description + instrumental/non-instrumental are supported via metadata fields and `/api/track_enrich`.
