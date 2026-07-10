from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from common import flatten_for_csv, get_project_root, load_yaml, save_json, timestamp_run_id, write_csv
from neural_common import ensure_neural_preflight, run_one_neural_experiment


def parse_seeds(seed_text: str) -> list[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def summarize_by_seed(summaries: list[dict[str, Any]], score_metric: str) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_seed.setdefault(int(summary["seed"]), []).append(summary)
    rows = []
    for seed, seed_summaries in sorted(by_seed.items()):
        scores = np.array([float(s[score_metric]) for s in seed_summaries], dtype=np.float64)
        means = np.array([float(s["overall_point_mean_error"]) for s in seed_summaries], dtype=np.float64)
        medians = np.array([float(s["overall_point_median_error"]) for s in seed_summaries], dtype=np.float64)
        rmses = np.array([float(s["overall_point_rmse_error"]) for s in seed_summaries], dtype=np.float64)
        p90s = np.array([float(s["overall_point_p90_error"]) for s in seed_summaries], dtype=np.float64)
        best_idx = int(np.argmin(scores))
        best_summary = seed_summaries[best_idx]
        rows.append(
            {
                "seed": int(seed),
                "num_repeats": int(len(seed_summaries)),
                "score_metric": score_metric,
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "mean_error_avg": float(np.mean(means)),
                "mean_error_std": float(np.std(means)),
                "median_error_avg": float(np.mean(medians)),
                "median_error_std": float(np.std(medians)),
                "rmse_error_avg": float(np.mean(rmses)),
                "rmse_error_std": float(np.std(rmses)),
                "p90_error_avg": float(np.mean(p90s)),
                "p90_error_std": float(np.std(p90s)),
                "best_repeat_run_id": best_summary["run_id"],
                "best_repeat_run_dir": best_summary["run_dir"],
                "best_repeat_score": float(best_summary[score_metric]),
            }
        )
    return rows


def run_neural_seed_search(analysis_config: dict[str, Any], seeds_override: list[int] | None = None, dry_run: bool = False) -> None:
    # Must run before the first main.py invocation of this sweep: NeuralWorker
    # needs project_root/output/neighbor_matrix.npy to already exist the
    # moment it starts (see neural_common.ensure_neural_preflight docstring).
    ensure_neural_preflight(analysis_config, dry_run=dry_run)

    neural_cfg = analysis_config.get("neural", {})
    eval_cfg = analysis_config.get("evaluation", {})
    seed_cfg = neural_cfg.get("seed_search", {})
    prefix = str(neural_cfg.get("run_name_prefix_seed_search", "neural_seed_search"))
    search_id = timestamp_run_id(prefix)
    seeds = seeds_override if seeds_override is not None else list(seed_cfg.get("seeds", []))
    if not seeds:
        raise ValueError("No seeds specified for neural seed search.")
    repeats_per_seed = int(seed_cfg.get("repeats_per_seed", 1))
    fixed_hyperparameters = seed_cfg.get("fixed_hyperparameters", {}) or {}
    if not isinstance(fixed_hyperparameters, dict):
        raise TypeError("neural.seed_search.fixed_hyperparameters must be a dictionary.")
    score_metric = str(eval_cfg.get("score_metric", "overall_point_mean_error"))
    summaries = []
    search_root = get_project_root() / analysis_config.get("project", {}).get("experiments_root", "experiments") / search_id
    for seed in seeds:
        for repeat_index in range(repeats_per_seed):
            run_label = f"seed_{int(seed):04d}_rep_{repeat_index:02d}"
            run_id = f"{search_id}/runs/{run_label}"
            print("\n" + "#" * 100)
            print(f"[Neural Seed Search] seed={seed}, repeat={repeat_index}")
            print(f"Fixed hyperparameters: {fixed_hyperparameters}")
            print("#" * 100)
            summary = run_one_neural_experiment(analysis_config, run_id, run_label, int(seed), dict(fixed_hyperparameters), dry_run=dry_run)
            if summary is None:
                continue
            summary.update(
                {
                    "search_id": search_id,
                    "search_type": "neural_seed_search",
                    "seed": int(seed),
                    "repeat_index": int(repeat_index),
                    "fixed_hyperparameters": json.dumps(fixed_hyperparameters, ensure_ascii=False),
                }
            )
            summaries.append(summary)
            write_csv([flatten_for_csv(s) for s in summaries], search_root / "neural_seed_search_summary_partial.csv")
    if dry_run:
        print("[Dry run] Seed search finished without executing main.py.")
        return
    if not summaries:
        raise RuntimeError("No successful neural seed-search runs.")
    seed_rows = summarize_by_seed(summaries, score_metric)
    ranked = sorted(seed_rows, key=lambda row: float(row["score_mean"]))
    write_csv([flatten_for_csv(s) for s in summaries], search_root / "neural_seed_search_all_runs.csv")
    write_csv([flatten_for_csv(s) for s in seed_rows], search_root / "neural_seed_search_by_seed.csv")
    write_csv([flatten_for_csv(s) for s in ranked], search_root / "neural_seed_search_ranked.csv")
    save_json({"all_runs": summaries, "by_seed": seed_rows, "ranked": ranked}, search_root / "neural_seed_search_summary.json")
    (search_root / "best_seed.txt").write_text(str(ranked[0]["seed"]), encoding="utf-8")
    save_json(ranked[0], search_root / "best_seed_result.json")
    print("\n" + "=" * 100)
    print("[Best Neural Seed]")
    print(f"score_metric={score_metric}")
    print(f"seed={ranked[0]['seed']}")
    print(f"score_mean={ranked[0]['score_mean']}")
    print(f"best_repeat_run_dir={ranked[0]['best_repeat_run_dir']}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", default="analysis/analysis_config.yaml")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, e.g. 0,1,2,18")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    analysis_config = load_yaml(get_project_root() / args.analysis_config)
    seeds_override = parse_seeds(args.seeds) if args.seeds is not None else None
    run_neural_seed_search(analysis_config, seeds_override=seeds_override, dry_run=args.dry_run)


if __name__ == "__main__":
    main()