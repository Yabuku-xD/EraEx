import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path


# Match run.py startup environment to avoid TensorFlow-side noise.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

sys.path.append(str(Path(__file__).parent.parent))

from config import settings  # noqa: E402


# Parse CLI arguments.
def _build_parser():
    """
    Build parser.
    
    This function implements the build parser step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Offline search hyperparameter tuning/evaluation for EraEx "
            "(Recall@K / NDCG@K over a labeled query set)."
        )
    )
    parser.add_argument(
        "queries_file",
        type=str,
        help="Path to labeled query set JSON/JSONL file.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,20",
        help="Comma-separated cutoffs for metrics (default: 5,10,20).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Random-search trials (default: 60). Use 0 for baseline-only evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible tuning.",
    )
    parser.add_argument(
        "--time-budget-sec",
        type=float,
        default=0.0,
        help="Optional wall-clock budget for tuning (0 disables).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Search result limit used for evaluation (must be >= max K). Default: 50.",
    )
    parser.add_argument(
        "--write-best",
        action="store_true",
        help="Write best overrides to settings.SEARCH_HPARAMS_PATH.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Override output path for best tuning JSON (default: settings.SEARCH_HPARAMS_PATH).",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional path to save full tuning report JSON.",
    )
    parser.add_argument(
        "--show-per-query",
        action="store_true",
        help="Print per-query metrics for baseline/best evaluations.",
    )
    parser.add_argument(
        "--template-out",
        type=str,
        default="",
        help="Write a sample query-set template JSON and exit.",
    )
    return parser


# Parse a comma list of positive integers.
def _parse_ks(text):
    """
    Parse ks.
    
    This function implements the parse ks step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    ks = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except Exception:
            continue
        if value > 0:
            ks.append(value)
    ks = sorted(set(ks))
    if not ks:
        ks = [5, 10, 20]
    return ks


# Write a sample labeled query-set template file.
def _write_template(path):
    """
    Execute write template.
    
    This function implements the write template step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    sample = {
        "queries": [
            {
                "query": "moody late night rnb like partynextdoor",
                "relevance": {
                    "wV2wBxqdZf0": 3,
                    "VVcKfQlzSCk": 2,
                    "k5TUDWZBuPM": 2,
                },
                "notes": "Use graded relevance 1-3 where 3 is the strongest match.",
            },
            {
                "query": "chill rnb 2016",
                "relevant_ids": ["VVcKfQlzSCk", "9pq93-A-vQU", "4PPHoCymPEA"],
            },
            {
                "query": "songs like PARTYNEXTDOOR but uplifting",
                "relevant_tracks": [
                    {"id": "VVcKfQlzSCk", "gain": 3},
                    {"id": "k5TUDWZBuPM", "gain": 1},
                ],
            },
        ]
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote query-set template: {target}")


# Load JSON or JSONL labeled queries.
def _load_queries_file(path):
    """
    Load queries file.
    
    This function implements the load queries file step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        return []
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = data.get("queries", [])
        if not isinstance(data, list):
            return []
        return data
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


# Normalize one labeled query row into {query, gains}.
def _normalize_query_row(row):
    """
    Normalize query row.
    
    This function implements the normalize query row step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not isinstance(row, dict):
        return None
    query = str(row.get("query", "") or "").strip()
    if not query:
        return None

    gains = {}

    relevance = row.get("relevance")
    if isinstance(relevance, dict):
        for key, value in relevance.items():
            tid = str(key or "").strip()
            if not tid:
                continue
            try:
                gain = float(value)
            except Exception:
                continue
            if gain > 0:
                gains[tid] = gain

    for key in ("relevant_ids", "positives"):
        vals = row.get(key)
        if isinstance(vals, list):
            for v in vals:
                tid = str(v or "").strip()
                if tid and tid not in gains:
                    gains[tid] = 1.0

    for key in ("relevant_tracks", "judgments"):
        vals = row.get(key)
        if isinstance(vals, list):
            for item in vals:
                if isinstance(item, dict):
                    tid = str(item.get("id", item.get("track_id", "")) or "").strip()
                    if not tid:
                        continue
                    try:
                        gain = float(item.get("gain", item.get("relevance", 1.0)))
                    except Exception:
                        gain = 1.0
                    if gain > 0:
                        gains[tid] = gain

    if not gains:
        return None
    return {"query": query, "gains": gains}


# Normalize and filter the full labeled query set.
def _normalize_query_set(rows):
    """
    Normalize query set.
    
    This function implements the normalize query set step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    normalized = []
    for row in rows or []:
        q = _normalize_query_row(row)
        if q is not None:
            normalized.append(q)
    return normalized


# Compute DCG@K from ranked ids and gain map.
def _dcg_at_k(ranked_ids, gains_by_id, k):
    """
    Execute dcg at k.
    
    This function implements the dcg at k step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    dcg = 0.0
    for idx, tid in enumerate((ranked_ids or [])[: int(k)], start=1):
        gain = float(gains_by_id.get(str(tid), 0.0))
        if gain <= 0.0:
            continue
        dcg += (2.0**gain - 1.0) / math.log2(idx + 1.0)
    return dcg


# Compute NDCG@K.
def _ndcg_at_k(ranked_ids, gains_by_id, k):
    """
    Execute ndcg at k.
    
    This function implements the ndcg at k step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    ideal_ids = [tid for tid, _ in sorted(gains_by_id.items(), key=lambda kv: kv[1], reverse=True)]
    ideal = _dcg_at_k(ideal_ids, gains_by_id, k)
    if ideal <= 0.0:
        return 0.0
    return _dcg_at_k(ranked_ids, gains_by_id, k) / ideal


# Compute Recall@K.
def _recall_at_k(ranked_ids, gains_by_id, k):
    """
    Execute recall at k.
    
    This function implements the recall at k step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    positives = {str(tid) for tid, gain in gains_by_id.items() if float(gain) > 0.0}
    if not positives:
        return 0.0
    hits = {str(tid) for tid in (ranked_ids or [])[: int(k)]}
    return len(positives.intersection(hits)) / float(len(positives))


# Macro-average metrics over a labeled query set.
def _evaluate_query_set(pipeline, labeled_queries, ks, limit, show_per_query=False):
    """
    Execute evaluate query set.
    
    This function implements the evaluate query set step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    metrics = {f"recall@{k}": [] for k in ks}
    metrics.update({f"ndcg@{k}": [] for k in ks})
    per_query = []

    for row in labeled_queries:
        query = row["query"]
        gains = row["gains"]
        payload = pipeline.search(query, limit=limit, offset=0)
        results = payload.get("results", []) if isinstance(payload, dict) else []
        ranked_ids = [str(r.get("id", "")) for r in results if isinstance(r, dict)]
        q_metrics = {"query": query, "num_relevant": len(gains), "num_results": len(ranked_ids)}
        for k in ks:
            r = _recall_at_k(ranked_ids, gains, k)
            n = _ndcg_at_k(ranked_ids, gains, k)
            metrics[f"recall@{k}"].append(r)
            metrics[f"ndcg@{k}"].append(n)
            q_metrics[f"recall@{k}"] = round(r, 6)
            q_metrics[f"ndcg@{k}"] = round(n, 6)
        per_query.append(q_metrics)
        if show_per_query:
            recall_cols = " ".join([f"R@{k}={q_metrics[f'recall@{k}']:.3f}" for k in ks])
            ndcg_cols = " ".join([f"N@{k}={q_metrics[f'ndcg@{k}']:.3f}" for k in ks])
            print(f"[Q] {query} | {recall_cols} | {ndcg_cols}")

    summary = {
        key: (sum(vals) / len(vals) if vals else 0.0)
        for key, vals in metrics.items()
    }
    return {"summary": summary, "per_query": per_query}


# Compute tuning objective from aggregated metrics.
def _objective_from_summary(summary, ks):
    """
    Execute objective from summary.
    
    This function implements the objective from summary step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    if not summary:
        return 0.0
    ndcg_key = f"ndcg@{10 if 10 in ks else max(ks)}"
    recall_key = f"recall@{20 if 20 in ks else max(ks)}"
    ndcg_val = float(summary.get(ndcg_key, 0.0))
    recall_val = float(summary.get(recall_key, 0.0))
    return (0.65 * ndcg_val) + (0.35 * recall_val)


# Clear caches that would otherwise hide weight changes between trials.
def _reset_pipeline_trial_state(pipeline):
    """
    Execute reset pipeline trial state.
    
    This function implements the reset pipeline trial state step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    try:
        pipeline._result_cache.clear()
    except Exception:
        pass
    try:
        pipeline._search_hparams_last_check = 0.0
    except Exception:
        pass


# Build an absolute override dict from multiplicative factors sampled in random search.
def _build_trial_overrides(rng):
    """
    Build trial overrides.
    
    This function implements the build trial overrides step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    def mul(lo, hi):
        """
        Execute mul.
        
        This function implements the mul step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return float(rng.uniform(lo, hi))

    return {
        "SEARCH_NON_LIKE_WEIGHTS": {
            "semantic": settings.SEARCH_NON_LIKE_WEIGHTS.get("semantic", 0.6) * mul(0.75, 1.35),
            "tag_overlap": settings.SEARCH_NON_LIKE_WEIGHTS.get("tag_overlap", 0.3) * mul(0.75, 1.45),
            "facet": settings.SEARCH_NON_LIKE_WEIGHTS.get("facet", 0.20) * mul(0.75, 1.50),
            "description": settings.SEARCH_NON_LIKE_WEIGHTS.get("description", 0.18) * mul(0.75, 1.50),
            "audio": settings.SEARCH_NON_LIKE_WEIGHTS.get("audio", 0.14) * mul(0.70, 1.50),
            "popularity": settings.SEARCH_NON_LIKE_WEIGHTS.get("popularity", 0.1) * mul(0.70, 1.30),
            "year_bonus": settings.SEARCH_NON_LIKE_WEIGHTS.get("year_bonus", 0.04) * mul(0.70, 1.50),
            "title_penalty": settings.SEARCH_NON_LIKE_WEIGHTS.get("title_penalty", 0.08) * mul(0.70, 1.45),
        },
        "SEARCH_LIKE_WEIGHTS": {
            "semantic": settings.SEARCH_LIKE_WEIGHTS.get("semantic", 0.42) * mul(0.75, 1.35),
            "overlap": settings.SEARCH_LIKE_WEIGHTS.get("overlap", 0.13) * mul(0.75, 1.45),
            "facet": settings.SEARCH_LIKE_WEIGHTS.get("facet", 0.30) * mul(0.75, 1.50),
            "artist": settings.SEARCH_LIKE_WEIGHTS.get("artist", 0.10) * mul(0.75, 1.60),
            "audio": settings.SEARCH_LIKE_WEIGHTS.get("audio", 0.06) * mul(0.70, 1.70),
            "popularity": settings.SEARCH_LIKE_WEIGHTS.get("popularity", 0.05) * mul(0.60, 1.30),
        },
        "SEARCH_FACET_WEIGHTS": {
            "genre": settings.SEARCH_FACET_WEIGHTS.get("genre", 0.40) * mul(0.75, 1.40),
            "vibe": settings.SEARCH_FACET_WEIGHTS.get("vibe", 0.35) * mul(0.75, 1.40),
            "mood": settings.SEARCH_FACET_WEIGHTS.get("mood", 0.25) * mul(0.75, 1.40),
        },
        "SEARCH_INSTRUMENTAL_WEIGHTS": {
            "match_boost": settings.SEARCH_INSTRUMENTAL_WEIGHTS.get("match_boost", 0.12) * mul(0.70, 1.60),
            "mismatch_penalty": settings.SEARCH_INSTRUMENTAL_WEIGHTS.get("mismatch_penalty", 0.18) * mul(0.70, 1.60),
            "unknown_penalty": settings.SEARCH_INSTRUMENTAL_WEIGHTS.get("unknown_penalty", 0.04) * mul(0.70, 1.40),
            "confidence_floor": settings.SEARCH_INSTRUMENTAL_WEIGHTS.get("confidence_floor", 0.35) * mul(0.85, 1.20),
        },
        "profiles": {
            "facet_heavy": {
                "SEARCH_NON_LIKE_WEIGHTS": {
                    "facet": settings.SEARCH_NON_LIKE_WEIGHTS.get("facet", 0.20) * mul(0.90, 1.80),
                    "description": settings.SEARCH_NON_LIKE_WEIGHTS.get("description", 0.18) * mul(0.90, 1.70),
                    "semantic": settings.SEARCH_NON_LIKE_WEIGHTS.get("semantic", 0.6) * mul(0.65, 1.15),
                    "audio": settings.SEARCH_NON_LIKE_WEIGHTS.get("audio", 0.14) * mul(0.80, 1.60),
                }
            },
            "like_mode": {
                "SEARCH_LIKE_WEIGHTS": {
                    "artist": settings.SEARCH_LIKE_WEIGHTS.get("artist", 0.10) * mul(0.85, 1.80),
                    "facet": settings.SEARCH_LIKE_WEIGHTS.get("facet", 0.30) * mul(0.85, 1.60),
                    "audio": settings.SEARCH_LIKE_WEIGHTS.get("audio", 0.06) * mul(0.70, 1.90),
                }
            },
        },
    }


# Print a compact metric summary line.
def _print_summary(prefix, summary, ks, objective):
    """
    Execute print summary.
    
    This function implements the print summary step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    cols = []
    for k in ks:
        cols.append(f"R@{k}={float(summary.get(f'recall@{k}', 0.0)):.4f}")
    for k in ks:
        cols.append(f"NDCG@{k}={float(summary.get(f'ndcg@{k}', 0.0)):.4f}")
    cols.append(f"OBJ={objective:.4f}")
    print(f"{prefix} | " + " | ".join(cols))


# Run baseline + random-search tuning and optionally write best overrides.
def main():
    """
    Run the command entry point.
    
    This function implements the main step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    args = _build_parser().parse_args()

    if args.template_out:
        _write_template(args.template_out)
        return

    from src.search.search_pipeline import UnifiedSearchPipeline  # noqa: E402

    ks = _parse_ks(args.ks)
    limit = max(int(args.limit), max(ks))
    query_rows = _load_queries_file(args.queries_file)
    labeled_queries = _normalize_query_set(query_rows)
    if not labeled_queries:
        raise SystemExit("No valid labeled queries found. See --template-out for the expected format.")

    original_hparams_path = getattr(settings, "SEARCH_HPARAMS_PATH", None)
    with tempfile.TemporaryDirectory(prefix="eraex_search_tuning_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "search_hparams_trial.json"
        settings.SEARCH_HPARAMS_PATH = tmp_path
        settings.SEARCH_DYNAMIC_WEIGHTS = True
        tmp_path.write_text("{}", encoding="utf-8")

        pipeline = UnifiedSearchPipeline()
        if pipeline.faiss_index is None:
            raise SystemExit("FAISS index not found. Build data/indexes first.")

        print(f"Loaded {len(labeled_queries)} labeled queries")
        print(f"Evaluation limit={limit}, ks={ks}")
        print(f"Temporary tuning overrides path: {tmp_path}")

        _reset_pipeline_trial_state(pipeline)
        baseline_eval = _evaluate_query_set(
            pipeline,
            labeled_queries,
            ks=ks,
            limit=limit,
            show_per_query=bool(args.show_per_query),
        )
        baseline_summary = baseline_eval["summary"]
        baseline_obj = _objective_from_summary(baseline_summary, ks)
        _print_summary("BASELINE", baseline_summary, ks, baseline_obj)

        best = {
            "objective": baseline_obj,
            "summary": baseline_summary,
            "overrides": {},
            "trial": -1,
        }
        history = [
            {
                "trial": -1,
                "objective": baseline_obj,
                "summary": baseline_summary,
                "overrides": {},
            }
        ]

        trial_count = max(0, int(args.trials))
        rng = random.Random(int(args.seed))
        started = time.time()

        for trial_idx in range(trial_count):
            if args.time_budget_sec and (time.time() - started) >= float(args.time_budget_sec):
                print(f"Time budget reached at trial {trial_idx}/{trial_count}")
                break

            overrides = _build_trial_overrides(rng)
            tmp_path.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")

            _reset_pipeline_trial_state(pipeline)
            evaluation = _evaluate_query_set(
                pipeline,
                labeled_queries,
                ks=ks,
                limit=limit,
                show_per_query=False,
            )
            summary = evaluation["summary"]
            objective = _objective_from_summary(summary, ks)
            history.append(
                {
                    "trial": trial_idx,
                    "objective": objective,
                    "summary": summary,
                    "overrides": overrides,
                }
            )
            print(
                f"[trial {trial_idx + 1}/{trial_count}] "
                f"obj={objective:.4f} "
                f"R@{ks[-1]}={summary.get(f'recall@{ks[-1]}', 0.0):.4f} "
                f"NDCG@{10 if 10 in ks else ks[-1]}="
                f"{summary.get(f'ndcg@{10 if 10 in ks else ks[-1]}', 0.0):.4f}"
            )

            if objective > float(best["objective"]):
                best = {
                    "objective": objective,
                    "summary": summary,
                    "overrides": overrides,
                    "trial": trial_idx,
                }
                print(f"  -> new best (trial {trial_idx + 1})")

        _print_summary("BEST", best["summary"], ks, best["objective"])

        if bool(args.show_per_query):
            if best["overrides"]:
                tmp_path.write_text(
                    json.dumps(best["overrides"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                _reset_pipeline_trial_state(pipeline)
            best_eval = _evaluate_query_set(
                pipeline,
                labeled_queries,
                ks=ks,
                limit=limit,
                show_per_query=True,
            )
            best["per_query"] = best_eval["per_query"]

        if args.write_best:
            output_path = Path(args.output_path) if args.output_path else Path(original_hparams_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(best["overrides"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Wrote best overrides to: {output_path}")

        if args.report_json:
            report = {
                "queries_file": str(Path(args.queries_file).resolve()),
                "ks": ks,
                "limit": limit,
                "baseline": {
                    "objective": baseline_obj,
                    "summary": baseline_summary,
                },
                "best": best,
                "num_trials_requested": trial_count,
                "num_trials_run": max(0, len(history) - 1),
                "history": history,
            }
            report_path = Path(args.report_json)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Wrote tuning report: {report_path}")

    settings.SEARCH_HPARAMS_PATH = original_hparams_path


if __name__ == "__main__":
    main()