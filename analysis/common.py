from __future__ import annotations

import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ============================================================
# Known upstream caveats (see README.md "Known upstream caveats"
# section for the full write-up). These are behaviors of main.py /
# the symbolic & neural workers that this analysis/ package works
# around from the outside, because those files are not part of this
# package:
#
# 1. neighbor_matrix.npy path mismatch:
#    SymbolicWorker (ProposedEstimatorStrategy.__init__) saves
#    neighbor_matrix.npy to a HARD-CODED "output/neighbor_matrix.npy"
#    path relative to the process cwd, ignoring config["OUTPUT_DIR"].
#    NeuralWorker._load_neighbor_index_matrix(), however, loads it
#    from config["OUTPUT_DIR"], which for every experiment run here is
#    a nested, run-specific folder. Because NeuralWorker.setup() needs
#    this file immediately (it can start before SymbolicWorker has
#    written anything), every run needs a valid copy pre-staged in its
#    own OUTPUT_DIR before main.py is invoked, and that copy needs an
#    original to come from somewhere. ensure_root_neighbor_matrix()
#    below creates that original once, at project_root/output/, by
#    running main.py directly from project_root (so the hard-coded
#    relative path lands exactly there). copy_neighbor_matrix_if_available()
#    then stages a copy of it into every run's OUTPUT_DIR.
#
# 2. GatingEvaluator checkpoint path is also hard-coded (relative
#    "checkpoint/<SCENARIO_NAME>.ckpt", ignoring config["CHECKPOINT_DIR"]),
#    and it runs its neural gating pass whenever aggregated_csi is not
#    None -- regardless of ENABLE_NEURAL. This can't be fixed from this
#    package without editing main.py itself; run_symbolic_ablation.py
#    documents the practical implication instead (see its module
#    docstring and the printed warning at the start of a run).
# ============================================================


def get_project_root() -> Path:
    # Expected location: analysis/<script>.py
    return Path(__file__).resolve().parents[1]


def timestamp_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def symlink_or_copy_file(src: Path, dst: Path, use_symlink: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def run_cmd_to_log(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool = False) -> None:
    print("\n" + "=" * 100)
    print("[RUN]")
    print(f"cwd: {cwd}")
    print(" ".join(str(c) for c in cmd))
    print(f"log: {log_path}")
    print("=" * 100)
    if dry_run:
        print("[Dry run] Command not executed.")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with return code {proc.returncode}. See log: {log_path}")


def load_base_and_merged_config(project_root: Path, base_config_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = project_root / base_config_path
    base_config = load_yaml(config_path)
    for key in ["DATASET_FOLDER", "ENV_CONFIG"]:
        if key not in base_config:
            raise KeyError(f"Missing required key {key!r} in {config_path}")
    original_dataset_folder = project_root / "dataset" / str(base_config["DATASET_FOLDER"])
    env_config = load_yaml(original_dataset_folder / str(base_config["ENV_CONFIG"]))
    merged = dict(base_config)
    merged.update(env_config)
    return base_config, merged


def create_run_workspace(project_root: Path, experiments_root: str, run_id: str, use_symlink: bool) -> dict[str, Path]:
    run_dir = project_root / experiments_root / run_id
    paths = {
        "run_dir": run_dir,
        "dataset_root": run_dir / "dataset",
        "output_root": run_dir / "output",
        "checkpoint_dir": run_dir / "checkpoint",
        "configs_dir": run_dir / "configs",
        "metrics_dir": run_dir / "metrics",
        "metadata_dir": run_dir / "metadata",
        "logs_dir": run_dir / "logs",
    }
    for p in paths.values():
        ensure_dir(p)
    directions_path = project_root / "directions.mat"
    if directions_path.exists():
        symlink_or_copy_file(directions_path, run_dir / "directions.mat", use_symlink)
    else:
        print(f"[Warning] directions.mat not found at {directions_path}. Continue anyway.")
    return paths


def snapshot_project_configs(project_root: Path, run_paths: dict[str, Path], base_config_path: str, base_config: dict[str, Any]) -> None:
    snapshot_dir = run_paths["metadata_dir"] / "config_snapshot"
    ensure_dir(snapshot_dir)
    copy_file(project_root / base_config_path, snapshot_dir / Path(base_config_path).name)
    original_dataset_folder = project_root / "dataset" / str(base_config["DATASET_FOLDER"])
    env_name = str(base_config["ENV_CONFIG"])
    copy_file(original_dataset_folder / env_name, snapshot_dir / env_name)


def prepare_experiment_dataset_env(project_root: Path, run_paths: dict[str, Path], base_config: dict[str, Any], use_symlink: bool) -> tuple[Path, Path]:
    original_dataset_folder = project_root / "dataset" / str(base_config["DATASET_FOLDER"])
    experiment_dataset_folder = run_paths["dataset_root"] / str(base_config["DATASET_FOLDER"])
    ensure_dir(experiment_dataset_folder)
    env_name = str(base_config["ENV_CONFIG"])
    symlink_or_copy_file(original_dataset_folder / env_name, experiment_dataset_folder / env_name, use_symlink)
    return original_dataset_folder, experiment_dataset_folder


def copy_neighbor_matrix_if_available(project_root: Path, target_output_dir: Path, use_symlink: bool) -> None:
    src = project_root / "output" / "neighbor_matrix.npy"
    if src.exists():
        symlink_or_copy_file(src, target_output_dir / "neighbor_matrix.npy", use_symlink)


def ensure_root_neighbor_matrix(
    project_root: Path,
    base_config: dict[str, Any],
    preflight_dataset: str,
    method: str,
    experiments_root: str,
    dry_run: bool = False,
) -> Path:
    """Guarantee that project_root/output/neighbor_matrix.npy exists.

    Why this is needed (see the module-level "Known upstream caveats" note
    above for the full explanation): NeuralWorker loads neighbor_matrix.npy
    from config["OUTPUT_DIR"] as soon as it starts, but SymbolicWorker only
    ever writes it to a hard-coded "output/neighbor_matrix.npy" relative to
    the process's cwd. copy_neighbor_matrix_if_available() bridges that gap
    by copying project_root/output/neighbor_matrix.npy into every run's own
    OUTPUT_DIR before main.py is invoked -- but that only works once an
    original copy exists at project_root/output/neighbor_matrix.npy.

    This function creates that original, once, by invoking main.py directly
    with cwd=project_root (so the hard-coded relative save path lands
    exactly at project_root/output/neighbor_matrix.npy), using a minimal
    symbolic-only pass: ENABLE_NEURAL=False and ENABLE_TRAJECTORY_DECODING=False,
    so no checkpoint or GPU training step is required for this to succeed.
    It is idempotent: if the file already exists, this is a no-op.

    A real dataset (one that already has cache.npy on disk) is required as
    input so that SymbolicWorker actually starts and constructs its
    estimator strategy; the neighbor_matrix save happens once, in that
    strategy's constructor, the moment the worker starts -- independent of
    how many blocks are ultimately in that dataset.
    """
    neighbor_path = project_root / "output" / "neighbor_matrix.npy"
    if neighbor_path.exists():
        return neighbor_path

    print("\n" + "!" * 100)
    print("[Preflight] project_root/output/neighbor_matrix.npy not found.")
    print("[Preflight] Generating it once via a minimal symbolic-only main.py pass")
    print(f"[Preflight] using dataset={preflight_dataset!r}, method={method!r}.")
    print("[Preflight] (Required because NeuralWorker cannot generate this file")
    print("[Preflight]  itself; see ensure_root_neighbor_matrix()'s docstring.)")
    print("!" * 100)

    if not preflight_dataset:
        raise ValueError(
            "Cannot run the neighbor_matrix preflight: no dataset was provided. "
            "Set dataset.train_datasets (or neural.preflight.dataset) in analysis_config.yaml."
        )

    cfg = dict(base_config)
    cfg["METHOD"] = method
    cfg["CSI_DATASETS"] = [preflight_dataset]
    cfg["ENABLE_NEURAL_TRAINING"] = False
    cfg["ENABLE_TRAJECTORY_DECODING"] = False
    # Match the hard-coded literal SymbolicWorker actually writes to, so the
    # only file we need out of this run lands exactly where later runs expect it.
    cfg["OUTPUT_DIR"] = "output"
    cfg["CHECKPOINT_DIR"] = "checkpoint"

    preflight_dir = project_root / experiments_root / "_preflight"
    ensure_dir(preflight_dir)
    config_path = preflight_dir / "neighbor_matrix_preflight_config.yaml"
    save_yaml(cfg, config_path)

    log_path = preflight_dir / "neighbor_matrix_preflight.log"
    cmd = [sys.executable, str(project_root / "main.py"), "--config", str(config_path)]

    # IMPORTANT: cwd=project_root (not a run-specific workspace) so that
    # SymbolicWorker's hard-coded relative "output/neighbor_matrix.npy" save
    # lands exactly at project_root/output/neighbor_matrix.npy.
    run_cmd_to_log(cmd, cwd=project_root, log_path=log_path, dry_run=dry_run)

    if dry_run:
        return neighbor_path

    if not neighbor_path.exists():
        raise FileNotFoundError(
            f"[Preflight] Failed to generate {neighbor_path}. "
            f"Check the preflight log for details: {log_path}"
        )

    print(f"[Preflight] neighbor_matrix.npy generated at: {neighbor_path}")
    return neighbor_path


def get_dataset_cache_path(dataset_folder: Path, dataset_name: str) -> Path:
    return dataset_folder / dataset_name / "cache.npy"


def load_cache(cache_path: Path) -> np.ndarray:
    if not cache_path.exists():
        raise FileNotFoundError(f"cache.npy not found: {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    if not isinstance(cache, np.ndarray) or cache.ndim < 1:
        raise ValueError(f"Invalid cache.npy at {cache_path}: {type(cache)}, shape={getattr(cache, 'shape', None)}")
    return cache


def materialize_full_cache_dataset(original_dataset_folder: Path, experiment_dataset_folder: Path, dataset_name: str, use_symlink: bool) -> str:
    source_cache = get_dataset_cache_path(original_dataset_folder, dataset_name)
    if not source_cache.exists():
        raise FileNotFoundError(f"cache.npy not found for dataset {dataset_name!r}: {source_cache}")
    target_dataset_dir = experiment_dataset_folder / dataset_name
    reset_dir(target_dataset_dir)
    symlink_or_copy_file(source_cache, target_dataset_dir / "cache.npy", use_symlink)
    return dataset_name


def build_segment_plan(total_blocks: int, segment_blocks: int, segment_stride: int, include_last_partial_segment: bool) -> list[dict[str, int]]:
    if segment_stride < segment_blocks:
        overlap = segment_blocks - segment_stride
        print(
            f"[Warning] segment_stride ({segment_stride}) < segment_blocks ({segment_blocks}): "
            f"consecutive segments will overlap by {overlap} block(s). "
            "Set segment_stride == segment_blocks in analysis_config.yaml if you need "
            "strictly non-overlapping segments."
        )
    segment_plan: list[dict[str, int]] = []
    segment_index = 0
    block_start = 0
    while block_start < total_blocks:
        remaining_blocks = total_blocks - block_start
        if remaining_blocks < segment_blocks and not include_last_partial_segment:
            break
        block_count = min(segment_blocks, remaining_blocks)
        segment_plan.append({"segment_index": segment_index, "block_start": block_start, "block_count": block_count})
        segment_index += 1
        block_start += segment_stride
    if not segment_plan:
        raise RuntimeError("No validation segment generated. Check segment settings and target cache length.")
    return segment_plan


def create_segment_cache_dataset(experiment_dataset_folder: Path, source_dataset_name: str, source_cache: np.ndarray, samples_per_block: int, segment_index: int, block_start: int, block_count: int) -> tuple[str, Path, dict[str, Any]]:
    sample_start = block_start * samples_per_block
    sample_end = sample_start + block_count * samples_per_block
    if sample_end > len(source_cache):
        raise ValueError(f"Segment exceeds cache length: sample_start={sample_start}, sample_end={sample_end}, cache_len={len(source_cache)}")
    segment_cache = source_cache[sample_start:sample_end]
    segment_dataset_name = f"{source_dataset_name}__seg_{segment_index:04d}_b{block_start:06d}"
    segment_dataset_dir = experiment_dataset_folder / segment_dataset_name
    reset_dir(segment_dataset_dir)
    np.save(segment_dataset_dir / "cache.npy", segment_cache, allow_pickle=True)
    meta = {
        "source_dataset": source_dataset_name,
        "segment_dataset_name": segment_dataset_name,
        "segment_index": int(segment_index),
        "block_start": int(block_start),
        "block_count": int(block_count),
        "samples_per_block": int(samples_per_block),
        "sample_start": int(sample_start),
        "sample_end": int(sample_end),
        "segment_cache_shape": list(segment_cache.shape),
        "segment_cache_dtype": str(segment_cache.dtype),
        "main_py_call_scope": "one_segment",
        "main_py_expected_internal_blocks": int(block_count),
    }
    save_json(meta, segment_dataset_dir / "segment_meta.json")
    return segment_dataset_name, segment_dataset_dir, meta


def make_main_config(base_config: dict[str, Any], csi_datasets: list[str], method: str, enable_neural_training: bool, enable_trajectory_decoding: bool, output_dir: str, checkpoint_dir: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(base_config)
    cfg["METHOD"] = method
    cfg["CSI_DATASETS"] = csi_datasets
    cfg["ENABLE_NEURAL_TRAINING"] = bool(enable_neural_training)
    cfg["ENABLE_TRAJECTORY_DECODING"] = bool(enable_trajectory_decoding)
    cfg["OUTPUT_DIR"] = output_dir
    # NOTE: checkpoint_dir is respected by NeuralWorker, but GatingEvaluator
    # inside ProposedEstimatorStrategy ignores it and always reads a
    # hard-coded relative "checkpoint/<SCENARIO_NAME>.ckpt" (see the
    # module-level "Known upstream caveats" note at the top of this file).
    # Passing a per-case checkpoint_dir here is still correct for the
    # neural training/validation path; it just does not achieve per-case
    # isolation for the symbolic gating checkpoint. See run_symbolic_ablation.py.
    cfg["CHECKPOINT_DIR"] = checkpoint_dir
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


def write_config(config: dict[str, Any], run_paths: dict[str, Path], filename: str) -> Path:
    config_path = run_paths["configs_dir"] / filename
    save_yaml(config, config_path)
    return config_path


def resolve_ground_truth_path(original_dataset_folder: Path, target_dataset: str, target_gt_paths: dict[str, str], explicit_gt_path: str | None, project_root: Path) -> Path:
    if explicit_gt_path is not None:
        path = Path(explicit_gt_path)
        return path if path.is_absolute() else project_root / path
    if target_dataset in target_gt_paths:
        path = Path(target_gt_paths[target_dataset])
        return path if path.is_absolute() else project_root / path
    return original_dataset_folder / "ground_truth" / f"{target_dataset}_gt.npy"


def normalize_xy_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[-1] >= 2:
        return arr[:, :2].astype(np.float64)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr.reshape(-1, arr.shape[-1])[:, :2].astype(np.float64)
    raise ValueError(f"Unsupported coordinate array shape: {arr.shape}")


def load_pred_trajectory(output_dataset_dir: Path) -> np.ndarray:
    for path in [output_dataset_dir / "trajectory.npy", output_dataset_dir / "predicted_trajectory.npy"]:
        if path.exists():
            return normalize_xy_array(np.load(path))
    raise FileNotFoundError(f"No trajectory output found in {output_dataset_dir}. Expected trajectory.npy or predicted_trajectory.npy.")


def select_ground_truth_segment(gt_all_xy: np.ndarray, block_start: int, block_count: int, num_sample: int, num_packet: int, samples_per_block: int, total_blocks: int, total_cache_samples: int) -> tuple[np.ndarray, str, int, int]:
    expected_traj_gt_len = total_blocks * num_sample
    if len(gt_all_xy) == expected_traj_gt_len:
        gt_start = block_start * num_sample
        gt_end = gt_start + block_count * num_sample
        return gt_all_xy[gt_start:gt_end], "trajectory_level", int(gt_start), int(gt_end)
    if len(gt_all_xy) == total_cache_samples:
        sample_start = block_start * samples_per_block
        sample_end = sample_start + block_count * samples_per_block
        raw_gt = gt_all_xy[sample_start:sample_end]
        expected_raw_len = block_count * num_sample * num_packet
        if len(raw_gt) == expected_raw_len:
            gt_btpx = raw_gt.reshape(block_count, num_sample, num_packet, 2)
            gt_btx = gt_btpx.mean(axis=2)
            return gt_btx.reshape(block_count * num_sample, 2), "raw_sample_level_packet_mean", int(sample_start), int(sample_end)
        return raw_gt, "raw_sample_level_no_reshape", int(sample_start), int(sample_end)
    print(
        f"[Warning] Ground truth length ({len(gt_all_xy)}) matches neither the "
        f"trajectory-level length (total_blocks*num_sample={expected_traj_gt_len}) "
        f"nor the raw sample-level length (total_cache_samples={total_cache_samples}). "
        "Falling back to trajectory-level slicing (gt_mode='fallback_trajectory_level'); "
        "double-check that this ground truth file actually corresponds to this target "
        "dataset, since alignment is not guaranteed in this branch."
    )
    gt_start = block_start * num_sample
    gt_end = gt_start + block_count * num_sample
    return gt_all_xy[gt_start:gt_end], "fallback_trajectory_level", int(gt_start), int(gt_end)


def compute_errors(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    n = min(len(pred_xy), len(gt_xy))
    if n <= 0:
        raise ValueError(f"Cannot compute errors: pred_len={len(pred_xy)}, gt_len={len(gt_xy)}")
    return np.linalg.norm(pred_xy[:n] - gt_xy[:n], axis=1)


def error_stats(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "p90": float(np.percentile(errors, 90)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


def evaluate_segment_output(output_dataset_dir: Path, gt_all_xy: np.ndarray, target_dataset: str, run_label: str, segment_index: int, block_start: int, block_count: int, num_sample: int, num_packet: int, samples_per_block: int, total_blocks: int, total_cache_samples: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    pred_xy = load_pred_trajectory(output_dataset_dir)
    gt_xy, gt_mode, gt_start, gt_end = select_ground_truth_segment(gt_all_xy, block_start, block_count, num_sample, num_packet, samples_per_block, total_blocks, total_cache_samples)
    errors = compute_errors(pred_xy, gt_xy)
    stats = error_stats(errors)
    row = {
        "run_label": run_label,
        "target_dataset": target_dataset,
        "segment_index": int(segment_index),
        "block_start": int(block_start),
        "block_count": int(block_count),
        "main_py_call_scope": "one_segment",
        "gt_mode": gt_mode,
        "gt_start": int(gt_start),
        "gt_end": int(gt_end),
        "pred_len": int(len(pred_xy)),
        "expected_pred_len": int(block_count * num_sample),
        "gt_len": int(len(gt_xy)),
        "compare_len": int(len(errors)),
        "mean_error": stats["mean"],
        "std_error": stats["std"],
        "median_error": stats["median"],
        "rmse_error": stats["rmse"],
        "p90_error": stats["p90"],
        "min_error": stats["min"],
        "max_error": stats["max"],
        "output_dataset_dir": str(output_dataset_dir),
    }
    return row, errors, pred_xy, gt_xy


def summarize_segments(rows: list[dict[str, Any]], all_errors: list[np.ndarray]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No segment metrics to summarize.")
    concat_errors = np.concatenate(all_errors, axis=0)
    overall = error_stats(concat_errors)
    segment_mean = np.array([r["mean_error"] for r in rows], dtype=np.float64)
    segment_std = np.array([r["std_error"] for r in rows], dtype=np.float64)
    segment_median = np.array([r["median_error"] for r in rows], dtype=np.float64)
    segment_rmse = np.array([r["rmse_error"] for r in rows], dtype=np.float64)
    segment_p90 = np.array([r["p90_error"] for r in rows], dtype=np.float64)
    return {
        "num_segments": int(len(rows)),
        "num_error_points": int(len(concat_errors)),
        "overall_point_mean_error": overall["mean"],
        "overall_point_std_error": overall["std"],
        "overall_point_median_error": overall["median"],
        "overall_point_rmse_error": overall["rmse"],
        "overall_point_p90_error": overall["p90"],
        "overall_point_min_error": overall["min"],
        "overall_point_max_error": overall["max"],
        "segment_mean_error_avg": float(np.mean(segment_mean)),
        "segment_mean_error_std": float(np.std(segment_mean)),
        "segment_std_error_avg": float(np.mean(segment_std)),
        "segment_std_error_std": float(np.std(segment_std)),
        "segment_median_error_avg": float(np.mean(segment_median)),
        "segment_median_error_std": float(np.std(segment_median)),
        "segment_rmse_error_avg": float(np.mean(segment_rmse)),
        "segment_rmse_error_std": float(np.std(segment_rmse)),
        "segment_p90_error_avg": float(np.mean(segment_p90)),
        "segment_p90_error_std": float(np.std(segment_p90)),
    }


def build_param_grid(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not search_space:
        return [{}]
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]