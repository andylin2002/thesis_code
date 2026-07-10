from __future__ import annotations

import argparse
import json
from typing import Any

from common import build_param_grid, flatten_for_csv, get_project_root, load_base_and_merged_config, load_yaml, save_json, timestamp_run_id, write_csv
from neural_common import ensure_neural_preflight, run_one_neural_experiment


def resolve_fixed_seed(analysis_config: dict[str, Any]) -> int:
    project_root = get_project_root()
    project_cfg = analysis_config.get("project", {})
    neural_cfg = analysis_config.get("neural", {})
    param_cfg = neural_cfg.get("param_search", {})
    fixed_seed = param_cfg.get("fixed_seed", None)
    if fixed_seed is not None:
        return int(fixed_seed)
    base_config_path = str(project_cfg.get("base_config_path", "config.yaml"))
    base_config, _ = load_base_and_merged_config(project_root, base_config_path)
    if "RANDOM_SEED" not in base_config:
        raise KeyError("No neural.param_search.fixed_seed was provided and RANDOM_SEED is missing from the main config.")
    return int(base_config["RANDOM_SEED"])


def run_neural_param_search(analysis_config: dict[str, Any], max_runs_override: int | None = None, dry_run: bool = False) -> None:
    # Must run before the first main.py invocation of this sweep: NeuralWorker
    # needs project_root/output/neighbor_matrix.npy to already exist the
    # moment it starts (see neural_common.ensure_neural_preflight docstring).
    ensure_neural_preflight(analysis_config, dry_run=dry_run)

    neural_cfg = analysis_config.get("neural", {})
    eval_cfg = analysis_config.get("evaluation", {})
    param_cfg = neural_cfg.get("param_search", {})
    prefix = str(neural_cfg.get("run_name_prefix_param_search", "neural_param_search"))
    search_id = timestamp_run_id(prefix)
    fixed_seed = resolve_fixed_seed(analysis_config)
    param_grid = build_param_grid(param_cfg.get("search_space", {}) or {})
    max_runs = max_runs_override if max_runs_override is not None else param_cfg.get("max_runs", None)
    if max_runs is not None:
        param_grid = param_grid[: int(max_runs)]
    score_metric = str(eval_cfg.get("score_metric", "overall_point_mean_error"))
    summaries = []
    search_root = get_project_root() / analysis_config.get("project", {}).get("experiments_root", "experiments") / search_id
    for idx, params in enumerate(param_grid, start=1):
        run_label = f"param_{idx:03d}_seed_{fixed_seed:04d}"
        run_id = f"{search_id}/runs/{run_label}"
        print("\n" + "#" * 100)
        print(f"[Neural Param Search] {idx}/{len(param_grid)} seed={fixed_seed}")
        print(f"Params: {params}")
        print("#" * 100)
        summary = run_one_neural_experiment(analysis_config, run_id, run_label, fixed_seed, params, dry_run=dry_run)
        if summary is None:
            continue
        summary.update({"search_id": search_id, "search_type": "neural_param_search", "param_index": idx, "seed": fixed_seed, "params": json.dumps(params, ensure_ascii=False)})
        summaries.append(summary)
        write_csv([flatten_for_csv(s) for s in summaries], search_root / "neural_param_search_summary_partial.csv")
    if dry_run:
        print("[Dry run] Param search finished without executing main.py.")
        return
    if not summaries:
        raise RuntimeError("No successful neural parameter-search runs.")
    ranked = sorted(summaries, key=lambda row: float(row[score_metric]))
    write_csv([flatten_for_csv(s) for s in summaries], search_root / "neural_param_search_summary.csv")
    write_csv([flatten_for_csv(s) for s in ranked], search_root / "neural_param_search_ranked.csv")
    save_json({"summaries": summaries, "ranked": ranked}, search_root / "neural_param_search_summary.json")
    save_json(ranked[0], search_root / "best_result.json")
    print("\n" + "=" * 100)
    print("[Best Neural Hyperparameters]")
    print(f"score_metric={score_metric}")
    print(f"score={ranked[0][score_metric]}")
    print(f"params={ranked[0]['params']}")
    print(f"run_dir={ranked[0]['run_dir']}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", default="analysis/analysis_config.yaml")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    analysis_config = load_yaml(get_project_root() / args.analysis_config)
    run_neural_param_search(analysis_config, max_runs_override=args.max_runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()