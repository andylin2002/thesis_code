from __future__ import annotations

"""
analyze_neural_boundary.py

Place this file in <project_root>/analysis/ and run it after:

    python analysis/build_neural_boundary.py

It reads experiments/neural_boundary/ and produces:

1. localization_performance.csv
2. reliability_alignment.csv
3. reliability_distribution.csv
4. localization_per_segment.csv
5. neural_boundary_summary.json
6. neural_boundary_tables.tex

All localization statistics are pooled point-wise statistics over the complete
held-out trajectory. Pearson and Spearman are computed over all pooled AP-time
entries. Top-1 AP match rate is computed over time indices.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURAL_BOUNDARY_DIR = PROJECT_ROOT / "experiments" / "neural_boundary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze experiments/neural_boundary arrays.")
    parser.add_argument("--input-dir", type=Path, default=NEURAL_BOUNDARY_DIR)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_required(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_xy(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].astype(np.float64)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr.reshape(-1, arr.shape[-1])[:, :2].astype(np.float64)
    raise ValueError(f"Unsupported trajectory shape: {arr.shape}")


def normalize_unit_mean_per_t(r_qt: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    r = np.asarray(r_qt, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"Expected reliability [Q,T], got {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("Reliability contains NaN or infinity.")
    if np.any(r < 0):
        raise ValueError(f"Reliability contains negative values; min={r.min()}")
    mean_t = r.mean(axis=0, keepdims=True)
    if np.any(mean_t <= eps):
        raise ValueError("Reliability has one or more zero-mean time columns.")
    return r / mean_t


def point_errors(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    pred = normalize_xy(pred_xy)
    gt = normalize_xy(gt_xy)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")
    return np.linalg.norm(pred - gt, axis=1)


def error_stats(errors: np.ndarray) -> dict[str, float]:
    e = np.asarray(errors, dtype=np.float64).reshape(-1)
    if e.size == 0:
        raise ValueError("Cannot summarize an empty error array.")
    return {
        "mean": float(np.mean(e)),
        "std": float(np.std(e)),
        "median": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "p90": float(np.percentile(e, 90)),
        "min": float(np.min(e)),
        "max": float(np.max(e)),
        "num_points": int(e.size),
    }


def pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Pearson shape mismatch: {x.shape} vs {y.shape}")
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denominator <= eps:
        return float("nan")
    return float(np.sum(x * y) / denominator)


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """Equivalent to scipy.stats.rankdata(method='average'), without SciPy."""
    x = np.asarray(x)
    sorter = np.argsort(x, kind="mergesort")
    inverse = np.empty_like(sorter)
    inverse[sorter] = np.arange(len(x))

    sorted_x = x[sorter]
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]

    ranks_sorted = np.empty(len(x), dtype=np.float64)
    for start, end in zip(starts, ends):
        # Rank origin does not affect Pearson correlation of ranks.
        ranks_sorted[start:end] = (start + end - 1) / 2.0
    return ranks_sorted[inverse]


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a).reshape(-1)
    y = np.asarray(b).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Spearman shape mismatch: {x.shape} vs {y.shape}")
    return pearson_corr(rankdata_average_ties(x), rankdata_average_ties(y))


def alignment_stats(a_qt: np.ndarray, b_qt: np.ndarray) -> dict[str, float]:
    if a_qt.shape != b_qt.shape:
        raise ValueError(f"Reliability shape mismatch: {a_qt.shape} vs {b_qt.shape}")
    return {
        "pearson": pearson_corr(a_qt, b_qt),
        "spearman": spearman_corr(a_qt, b_qt),
        "top1_match": float(
            np.mean(np.argmax(a_qt, axis=0) == np.argmax(b_qt, axis=0))
        ),
    }


def distribution_stats(r_qt: np.ndarray) -> dict[str, float]:
    r = normalize_unit_mean_per_t(r_qt)
    flat = r.reshape(-1)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }


def load_arrays(input_dir: Path) -> dict[str, np.ndarray]:
    arrays = {
        "ground_truth": normalize_xy(load_required(input_dir / "ground_truth_concat.npy")),
        "trajectory_uniform": normalize_xy(
            load_required(input_dir / "trajectory_uniform_concat.npy")
        ),
        "trajectory_neural": normalize_xy(
            load_required(input_dir / "trajectory_neural_concat.npy")
        ),
        "trajectory_emission_oracle": normalize_xy(
            load_required(input_dir / "trajectory_emission_oracle_concat.npy")
        ),
        "trajectory_decoding_oracle": normalize_xy(
            load_required(input_dir / "trajectory_decoding_oracle_concat.npy")
        ),
        "reliability_neural": normalize_unit_mean_per_t(
            load_required(input_dir / "reliability_neural_concat.npy")
        ),
        "reliability_emission_oracle": normalize_unit_mean_per_t(
            load_required(input_dir / "reliability_emission_oracle_concat.npy")
        ),
        "reliability_decoding_oracle": normalize_unit_mean_per_t(
            load_required(input_dir / "reliability_decoding_oracle_concat.npy")
        ),
        "segment_offsets": load_required(input_dir / "segment_offsets.npy").astype(np.int64),
        "segment_lengths": load_required(input_dir / "segment_lengths.npy").astype(np.int64),
    }

    gt_shape = arrays["ground_truth"].shape
    for key in (
        "trajectory_uniform",
        "trajectory_neural",
        "trajectory_emission_oracle",
        "trajectory_decoding_oracle",
    ):
        if arrays[key].shape != gt_shape:
            raise ValueError(f"{key} shape {arrays[key].shape} != GT shape {gt_shape}")

    q, t_total = arrays["reliability_neural"].shape
    if t_total != gt_shape[0]:
        raise ValueError(
            f"Reliability T={t_total} does not equal trajectory T={gt_shape[0]}."
        )
    for key in (
        "reliability_emission_oracle",
        "reliability_decoding_oracle",
    ):
        if arrays[key].shape != (q, t_total):
            raise ValueError(f"{key} shape mismatch: {arrays[key].shape} vs {(q, t_total)}")

    offsets = arrays["segment_offsets"]
    lengths = arrays["segment_lengths"]
    if offsets.ndim != 1 or lengths.ndim != 1 or len(offsets) != len(lengths):
        raise ValueError(
            f"Invalid segment metadata shapes: offsets={offsets.shape}, lengths={lengths.shape}"
        )
    if int(lengths.sum()) != t_total:
        raise ValueError(
            f"Segment lengths sum to {int(lengths.sum())}, expected total T={t_total}."
        )

    return arrays


def build_localization_table(arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any]]:
    gt = arrays["ground_truth"]
    settings = [
        ("Uniform", arrays["trajectory_uniform"]),
        ("Neural reliability", arrays["trajectory_neural"]),
        ("Emission oracle", arrays["trajectory_emission_oracle"]),
        ("Decoding oracle", arrays["trajectory_decoding_oracle"]),
    ]

    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for name, trajectory in settings:
        stats = error_stats(point_errors(trajectory, gt))
        raw[name] = stats
        rows.append(
            {
                "Weighting setting": name,
                "Mean (m)": round(stats["mean"], 3),
                "Median (m)": round(stats["median"], 3),
                "RMSE (m)": round(stats["rmse"], 3),
                "90th percentile (m)": round(stats["p90"], 3),
            }
        )
    return pd.DataFrame(rows), raw


def build_alignment_table(arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any]]:
    neural = arrays["reliability_neural"]
    emission = arrays["reliability_emission_oracle"]
    decoding = arrays["reliability_decoding_oracle"]

    pairs = [
        ("Neural reliability vs emission oracle", neural, emission),
        ("Neural reliability vs decoding oracle", neural, decoding),
        ("Emission oracle vs decoding oracle", emission, decoding),
    ]

    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for name, a, b in pairs:
        stats = alignment_stats(a, b)
        raw[name] = stats
        rows.append(
            {
                "Alignment": name,
                "Pearson": round(stats["pearson"], 3),
                "Spearman": round(stats["spearman"], 3),
                "Top-1 AP match rate": round(stats["top1_match"], 3),
            }
        )
    return pd.DataFrame(rows), raw


def build_distribution_table(arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any]]:
    distributions = [
        ("Neural reliability", arrays["reliability_neural"]),
        ("Emission oracle", arrays["reliability_emission_oracle"]),
        ("Decoding oracle", arrays["reliability_decoding_oracle"]),
    ]

    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for name, reliability in distributions:
        stats = distribution_stats(reliability)
        raw[name] = stats
        rows.append(
            {
                "Distribution": name,
                "Min": round(stats["min"], 3),
                "Max": round(stats["max"], 3),
                "Mean": round(stats["mean"], 3),
                "Std. dev.": round(stats["std"], 3),
            }
        )
    return pd.DataFrame(rows), raw


def build_segment_table(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    offsets = arrays["segment_offsets"]
    lengths = arrays["segment_lengths"]
    gt = arrays["ground_truth"]

    settings = [
        ("Uniform", arrays["trajectory_uniform"]),
        ("Neural reliability", arrays["trajectory_neural"]),
        ("Emission oracle", arrays["trajectory_emission_oracle"]),
        ("Decoding oracle", arrays["trajectory_decoding_oracle"]),
    ]

    rows: list[dict[str, Any]] = []
    cursor = 0
    for segment_index, (offset, length) in enumerate(zip(offsets, lengths)):
        start = cursor
        end = cursor + int(length)
        for name, trajectory in settings:
            stats = error_stats(point_errors(trajectory[start:end], gt[start:end]))
            rows.append(
                {
                    "Segment index": int(segment_index),
                    "Block offset": int(offset),
                    "Weighting setting": name,
                    "Number of points": int(length),
                    "Mean (m)": stats["mean"],
                    "Median (m)": stats["median"],
                    "RMSE (m)": stats["rmse"],
                    "90th percentile (m)": stats["p90"],
                }
            )
        cursor = end
    return pd.DataFrame(rows)


def latex_table(df: pd.DataFrame, column_format: str) -> str:
    return df.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        float_format=lambda x: f"{x:.3f}",
    )


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    arrays = load_arrays(input_dir)

    localization_df, localization_raw = build_localization_table(arrays)
    alignment_df, alignment_raw = build_alignment_table(arrays)
    distribution_df, distribution_raw = build_distribution_table(arrays)
    segment_df = build_segment_table(arrays)

    localization_df.to_csv(input_dir / "localization_performance.csv", index=False)
    alignment_df.to_csv(input_dir / "reliability_alignment.csv", index=False)
    distribution_df.to_csv(input_dir / "reliability_distribution.csv", index=False)
    segment_df.to_csv(input_dir / "localization_per_segment.csv", index=False)

    summary = {
        "input_dir": str(input_dir),
        "shape": {
            "ground_truth": list(arrays["ground_truth"].shape),
            "reliability": list(arrays["reliability_neural"].shape),
            "num_segments": int(len(arrays["segment_offsets"])),
        },
        "definitions": {
            "localization_statistics": "pooled point-wise errors over all matched segments",
            "pearson_spearman": "computed over all pooled AP-time entries",
            "top1_match": "fraction of time indices with the same highest-weight AP",
            "distribution_normalization": "each time column is normalized to mean 1 across APs",
        },
        "localization_performance": localization_raw,
        "reliability_alignment": alignment_raw,
        "reliability_distribution": distribution_raw,
    }
    save_json(summary, input_dir / "neural_boundary_summary.json")

    latex = (
        "% Localization performance\n"
        + latex_table(localization_df, "lrrrr")
        + "\n% Reliability alignment\n"
        + latex_table(alignment_df, "lrrr")
        + "\n% Reliability distribution\n"
        + latex_table(distribution_df, "lrrrr")
    )
    (input_dir / "neural_boundary_tables.tex").write_text(latex, encoding="utf-8")

    print("\n===== Localization performance =====")
    print(localization_df.to_string(index=False))

    print("\n===== Reliability alignment =====")
    print(alignment_df.to_string(index=False))

    print("\n===== Reliability distribution statistics =====")
    print(distribution_df.to_string(index=False))

    print(f"\n[Done] Tables, CSVs, and JSON were saved under: {input_dir}")


if __name__ == "__main__":
    main()
