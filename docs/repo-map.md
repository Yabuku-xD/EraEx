# EraEx Repository Map

This is a practical guide to where things live and when to open them.

## Start Here

- `run.py`: best first file if you want the runtime entrypoint.
- `README.md`: product-level overview, setup, and key workflows.
- `docs/architecture.md`: system design and request flow explanation.
- `config/settings.py`: runtime flags and scoring knobs.

## Root Directories

### `assets/`

Repository images used in the README and GitHub presentation.

### `cli_tools/`

Operational scripts grouped by purpose:

- `project_maintenance_cli.py`: main maintenance entrypoint.
- `search_query_cli.py`: inspect search behavior from the terminal.
- `foryou_recommendation_cli.py`: inspect recommendation behavior from the
  terminal.
- `search_tuning_cli.py`: offline search tuning/evaluation.
- `bulk_validation_cli.py`: larger validation runs.
- `get_query_validation.py`: point lookup into tracked validation output.
- `benchmark_optimized_dpp.py`, `retrieval_optimizer.py`,
  `parallel_retrieval_optimizer.py`, `ultrafast_quality_optimizer.py`,
  `foryou_optimizer.py`: optimization and experiment helpers.
- `view_bot_profiles.py`: inspect synthetic validation profiles.

### `config/`

- `settings.py`: central constants, environment parsing, retrieval weights,
  cache sizes, and feature flags.

### `data/`

Mixed runtime and tracked data:

- `eval/`: tracked evaluation plans and outputs.
- `users.db`: local SQLite store for accounts and library state.
- `indexes/` and `cache/`: expected runtime artifacts, usually generated and
  kept out of Git.

### `notebooks/`

- `full_pipeline.ipynb`: end-to-end rebuild workflow.
- `reembed_reindex_colab.ipynb`: remote or Colab rebuild path.
- `build_ultrafast_ivf_index.py`: faster ANN index generation helper.
- `eda_final.ipynb`: analysis notebook.

## Source Tree

### `src/core/`

Cross-cutting infrastructure:

- `text_embeddings.py`: sentence-transformer loading and encoding.
- `media_metadata.py`: media resolution, cover fallback, enrichment.
- `lyrics_provider_engine.py`: lyrics provider logic.
- `audio_processing.py`: audio extraction and steering helpers.
- `lazy_loading.py`: singleton-style lazy initialization utilities.

### `src/search/`

- `search_pipeline.py`: the main search engine and the highest-complexity file
  in the repo.

Open this when you need:

- query understanding logic
- FAISS retrieval behavior
- hybrid ranking changes
- search diagnostics or intent bugs

### `src/recommendation/`

- `recommendation_engine.py`: cold-start and adaptive ranking.
- `fast_dpp.py`: optimized diversity utilities.

Open this when you need:

- For You ranking logic
- cold-start behavior
- diversity selection
- recommendation latency or quality tuning

### `src/user_profiles/`

- `user_profile_store.py`: SQLite schema, migrations, likes, plays, skips,
  playlists, accounts.

### `src/web_api/`

- `web_app.py`: Flask app creation plus nearly all API routes and orchestration.

Open this when you need:

- endpoint behavior
- request/response contracts
- auth/session flows
- search/recommendation route wiring

### `src/templates/`

Server-rendered HTML shell and partials:

- `index.html`: top-level page composition
- `partials/view_*.html`: major app views
- modal, player, header, sidebar, and right-panel partials

### `src/static/`

- `app.js`: front-end JS manifest
- `app.css`: CSS manifest
- `js/app.core.js`: main client runtime
- `styles/*.css`: split styling by area

## Read Order By Goal

### Understand the product

1. `README.md`
2. `docs/architecture.md`
3. `src/web_api/web_app.py`
4. `src/static/js/app.core.js`

### Debug search quality

1. `config/settings.py`
2. `src/search/search_pipeline.py`
3. `cli_tools/search_query_cli.py`
4. `metrics.md`

### Debug recommendation quality

1. `src/recommendation/recommendation_engine.py`
2. `src/recommendation/fast_dpp.py`
3. `cli_tools/foryou_recommendation_cli.py`
4. `metrics.md`

### Rebuild data artifacts

1. `notebooks/full_pipeline.ipynb`
2. `src/pipeline/maintenance_tasks.py`
3. `cli_tools/project_maintenance_cli.py`

## Repository Hygiene Notes

- `__pycache__` and `.pyc` files should stay out of Git.
- `.env` is ignored and should remain local.
- Large index artifacts belong in generated data directories, not the repo.
- When docs change, prefer updating `README.md` plus one focused doc page
  instead of scattering explanation across multiple ad hoc markdown files.
