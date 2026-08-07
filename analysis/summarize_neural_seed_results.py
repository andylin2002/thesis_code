from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 每一項都是單一 seed 已經對全部 evaluation points 計算完成的結果。
METRICS: dict[str, str] = {
    "overall_point_mean_error": "Mean error",
    "overall_point_std_error": "Point-error std",
    "overall_point_median_error": "Median error",
    "overall_point_rmse_error": "RMSE",
    "overall_point_p90_error": "P90 error",
    "overall_point_min_error": "Minimum error",
    "overall_point_max_error": "Maximum error",
}

PRIMARY_METRICS = [
    "overall_point_mean_error",
    "overall_point_median_error",
    "overall_point_rmse_error",
    "overall_point_p90_error",
]


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_seed_search(
    experiments_root: Path,
    prefix: str,
) -> Path:
    """
    Find the most recently modified seed-search directory that contains
    neural_seed_search_summary.json.
    """

    if not experiments_root.is_dir():
        raise FileNotFoundError(
            f"Experiments root not found: {experiments_root}"
        )

    candidates = [
        path
        for path in experiments_root.iterdir()
        if (
            path.is_dir()
            and path.name.startswith(prefix)
            and (path / "neural_seed_search_summary.json").is_file()
        )
    ]

    if not candidates:
        raise FileNotFoundError(
            "No completed neural seed-search directory was found under "
            f"{experiments_root} with prefix '{prefix}'."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_search_dir(
    search_dir_text: str | None,
    experiments_root_text: str,
    prefix: str,
) -> Path:
    if search_dir_text is not None:
        search_dir = Path(search_dir_text)

        if not search_dir.is_absolute():
            search_dir = PROJECT_ROOT / search_dir

        search_dir = search_dir.resolve()

        if not search_dir.is_dir():
            raise FileNotFoundError(
                f"Seed-search directory not found: {search_dir}"
            )

        return search_dir

    experiments_root = Path(experiments_root_text)

    if not experiments_root.is_absolute():
        experiments_root = PROJECT_ROOT / experiments_root

    return find_latest_seed_search(
        experiments_root=experiments_root.resolve(),
        prefix=prefix,
    )


def load_seed_runs(search_dir: Path) -> list[dict[str, Any]]:
    summary_path = search_dir / "neural_seed_search_summary.json"
    summary_data = load_json(summary_path)

    if not isinstance(summary_data, dict):
        raise TypeError(
            f"Expected a JSON object in {summary_path}, "
            f"but received {type(summary_data).__name__}."
        )

    runs = summary_data.get("all_runs")

    if not isinstance(runs, list) or not runs:
        raise ValueError(
            f"No non-empty 'all_runs' list was found in {summary_path}."
        )

    validated_runs: list[dict[str, Any]] = []

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise TypeError(
                f"all_runs[{index}] is not a dictionary."
            )

        if "seed" not in run:
            raise KeyError(
                f"all_runs[{index}] does not contain 'seed'."
            )

        missing_metrics = [
            metric_name
            for metric_name in METRICS
            if metric_name not in run
        ]

        if missing_metrics:
            raise KeyError(
                f"Seed {run['seed']} is missing metrics: "
                f"{missing_metrics}"
            )

        validated_runs.append(run)

    validated_runs.sort(key=lambda run: int(run["seed"]))

    seeds = [int(run["seed"]) for run in validated_runs]

    if len(seeds) != len(set(seeds)):
        duplicates = sorted(
            seed
            for seed in set(seeds)
            if seeds.count(seed) > 1
        )

        raise ValueError(
            "Duplicate seeds were found even though repeats_per_seed is "
            f"expected to be 1: {duplicates}"
        )

    return validated_runs


def sample_std(values: np.ndarray) -> float:
    """
    Sample standard deviation across seeds.

    ddof=1 is used because the tested seeds are treated as samples of the
    variability induced by random initialization and stochastic training.
    """

    if values.size < 2:
        return 0.0

    return float(np.std(values, ddof=1))


def summarize_across_seeds(
    runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not runs:
        raise ValueError("Cannot summarize an empty run list.")

    seeds = np.asarray(
        [int(run["seed"]) for run in runs],
        dtype=np.int64,
    )

    metric_statistics: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []

    for metric_name, display_name in METRICS.items():
        values = np.asarray(
            [float(run[metric_name]) for run in runs],
            dtype=np.float64,
        )

        best_index = int(np.argmin(values))
        worst_index = int(np.argmax(values))

        statistics = {
            "display_name": display_name,
            "num_seeds": int(values.size),
            "across_seed_mean": float(np.mean(values)),
            "across_seed_sample_std": sample_std(values),
            "across_seed_median": float(np.median(values)),
            "across_seed_min": float(np.min(values)),
            "across_seed_max": float(np.max(values)),
            "best_seed": int(seeds[best_index]),
            "best_value": float(values[best_index]),
            "worst_seed": int(seeds[worst_index]),
            "worst_value": float(values[worst_index]),
        }

        metric_statistics[metric_name] = statistics

        csv_rows.append(
            {
                "metric": metric_name,
                **statistics,
            }
        )

    result = {
        "source_type": "neural_seed_search",
        "aggregation_unit": "seed",
        "repeat_per_seed": 1,
        "num_seeds": int(len(runs)),
        "seeds": [int(seed) for seed in seeds],
        "standard_deviation_definition": (
            "Sample standard deviation across seed-level metrics "
            "(numpy.std with ddof=1)."
        ),
        "metric_statistics": metric_statistics,
    }

    return result, csv_rows


def build_per_seed_rows(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for run in runs:
        row: dict[str, Any] = {
            "seed": int(run["seed"]),
            "run_id": run.get("run_id", ""),
            "run_dir": run.get("run_dir", ""),
        }

        for metric_name in METRICS:
            row[metric_name] = float(run[metric_name])

        rows.append(row)

    return rows


def print_summary(statistics: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("[Neural Seed Statistics]")
    print(f"num_seeds={statistics['num_seeds']}")
    print(f"seeds={statistics['seeds']}")
    print("-" * 100)

    metric_statistics = statistics["metric_statistics"]

    for metric_name in PRIMARY_METRICS:
        metric = metric_statistics[metric_name]

        print(
            f"{metric['display_name']:<14} "
            f"{metric['across_seed_mean']:.6f} "
            f"± {metric['across_seed_sample_std']:.6f} m "
            f"[min={metric['across_seed_min']:.6f}, "
            f"max={metric['across_seed_max']:.6f}]"
        )

    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute statistics across all seeds in one completed neural "
            "seed-search experiment."
        )
    )

    parser.add_argument(
        "--search-dir",
        default=None,
        help=(
            "Seed-search root directory. It may be absolute or relative "
            "to the project root. If omitted, the latest matching search "
            "directory under --experiments-root is used."
        ),
    )

    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help=(
            "Experiments directory used when --search-dir is omitted. "
            "Default: experiments"
        ),
    )

    parser.add_argument(
        "--prefix",
        default="neural_seed_search",
        help=(
            "Directory-name prefix used to find the latest seed search. "
            "Default: neural_seed_search"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. If omitted, statistics are saved "
            "inside the seed-search root."
        ),
    )

    args = parser.parse_args()

    search_dir = resolve_search_dir(
        search_dir_text=args.search_dir,
        experiments_root_text=args.experiments_root,
        prefix=args.prefix,
    )

    if args.output_dir is None:
        output_dir = search_dir
    else:
        output_dir = Path(args.output_dir)

        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        output_dir = output_dir.resolve()

    print(f"[Input seed-search directory] {search_dir}")

    runs = load_seed_runs(search_dir)

    statistics, metric_rows = summarize_across_seeds(runs)
    statistics["source_directory"] = str(search_dir)

    per_seed_rows = build_per_seed_rows(runs)

    save_json(
        statistics,
        output_dir / "neural_seed_across_seed_statistics.json",
    )

    write_csv(
        metric_rows,
        output_dir / "neural_seed_across_seed_statistics.csv",
    )

    write_csv(
        per_seed_rows,
        output_dir / "neural_seed_per_seed_metrics.csv",
    )

    print_summary(statistics)

    print("\nSaved:")
    print(
        output_dir
        / "neural_seed_across_seed_statistics.json"
    )
    print(
        output_dir
        / "neural_seed_across_seed_statistics.csv"
    )
    print(
        output_dir
        / "neural_seed_per_seed_metrics.csv"
    )


if __name__ == "__main__":
    main()