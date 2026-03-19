# EraEx Architecture

This document is the quickest way to understand how EraEx is put together and
how requests flow through the system.

## System View

```text
                         +----------------------+
                         | notebooks/           |
                         | cli_tools/           |
                         | rebuild + validate   |
                         +----------+-----------+
                                    |
                                    v
 +--------------------+    +----------------------+    +-------------------+
 | data/indexes/*     | -> | src/search/          | <- | src/core/         |
 | metadata + FAISS   |    | semantic retrieval   |    | embeddings/media  |
 +--------------------+    | candidate union      |    | lyrics/audio      |
                           | reranking + metrics  |    +-------------------+
                           +----------+-----------+
                                      |
                                      v
 +--------------------+    +----------------------+    +-------------------+
 | data/users.db      | -> | src/recommendation/  | -> | src/web_api/      |
 | likes, plays, auth |    | cold start + adapt   |    | Flask routes      |
 +--------------------+    +----------+-----------+    | orchestration     |
                                      |                +---------+---------+
                                      v                          |
                           +----------------------+              v
                           | src/templates/       |    +-------------------+
                           | src/static/          |    | Browser UI         |
                           | server-rendered UI   |    | Home/Search/Feed   |
                           +----------------------+    +-------------------+
```

## Main Execution Paths

### App Startup

`run.py` is the entrypoint. It:

1. Forces a PyTorch-only transformer path.
2. Imports the Flask app plus the search and recommendation loaders.
3. Warms the search pipeline, embedding model, and recommender.
4. Starts Flask once core systems are ready.

### Search Request

`GET /search` and `GET /sonic` both route into
`src/search/search_pipeline.py`, but use different modes.

- `/search`: hybrid mode for explicit search behavior.
- `/sonic`: semantic mode for discovery-style prompts.

The search pipeline is multi-stage:

1. Normalize and classify the query.
2. Detect intent such as artist-only, title-by-artist, like-mode, year, or
   instrumental preference.
3. Build candidates from semantic retrieval plus lexical and artist-aware
   recall channels.
4. Compute candidate features such as semantic match, facet overlap, title
   exactness, audio alignment, popularity, and utility-track penalties.
5. Rerank, diversify where appropriate, and attach query diagnostics.

### Recommendation Request

`GET /api/recommend` uses `src/recommendation/recommendation_engine.py`.

Two major modes exist:

- Cold start: used when a profile has little or no personal history.
- Adaptive personalization: used when likes, plays, playlists, or skips exist.

The recommendation engine combines:

- popularity priors
- similarity to recent positive signals
- penalties from dislikes and skips
- long-tail and diversity balancing
- DPP or fast diversity selection for result spread

### Metadata And Playback

`src/core/media_metadata.py` handles:

- cover fallback behavior
- YouTube resolution and retry candidates
- enrichment of description and instrumental metadata
- optional Spotify cover/playback support

`/api/resolve_video` and `/api/track_enrich` expose this behavior to the UI.

### User State

`src/user_profiles/user_profile_store.py` keeps local user state in SQLite:

- accounts
- likes and dislikes
- play history
- skip events
- playlists and playlist tracks

That makes EraEx usable as a local product without adding an external database.

## Module Responsibilities

### `src/web_api/`

- Flask routes
- auth session management
- recommendation/search orchestration
- API response shaping
- media and playlist endpoints

### `src/search/`

- query understanding
- FAISS retrieval
- candidate union and lexical recall
- hybrid reranking
- search metrics and diagnostics

### `src/recommendation/`

- popularity and catalog indexing
- cold-start selection
- adaptive personalization
- DPP and fast diversity utilities
- recommendation quotas and fallback behavior

### `src/core/`

- embedding model access
- lazy loaders
- media metadata and lyrics providers
- audio feature extraction

### `src/templates/` and `src/static/`

- server-rendered HTML shell
- modular CSS manifests
- single-page style client logic for search, queueing, player, auth, and
  playlists

## Data Contracts

### Required But Usually Ignored

- `data/indexes/metadata.json`
- `data/indexes/faiss_index.bin` or `faiss_index_ivf.bin`
- `data/indexes/id_map.json` or `track_ids.pkl`

These assets are large and generated outside normal source control flow.

### Tracked Support Data

- `data/eval/*`: evaluation plans and answer sets
- `metrics.md`: summary metrics for the current system snapshot
- `data/users.db`: local profile store

## Operational Surfaces

### CLI Layer

The repo now contains several CLI categories:

- maintenance: rebuild, enrich, clean, simulate
- debugging: query inspection and For You inspection
- validation: bulk validation and point lookup scripts
- optimization experiments: retrieval, parallelism, IVF, and fast DPP helpers

### Notebook Layer

Notebooks remain the heaviest maintenance surface for index rebuilding and
analysis. They are complementary to the CLI layer rather than replacements.

## Current Engineering Notes

- Search behavior is highly configurable through `config/settings.py`, which
  makes experimentation easy but increases the need for disciplined docs.
- Recommendation quality depends on external index artifacts; without them,
  compile-time checks still pass but runtime fidelity drops sharply.
- The repo currently has validation assets but no automated `pytest` suite.
