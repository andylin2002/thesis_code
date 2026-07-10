from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    build_segment_plan,
    copy_neighbor_matrix_if_available,
    create_run_workspace,
    create_segment_cache_dataset,
    ensure_dir,
    evaluate_segment_output,
    get_dataset_cache_path,
    get_project_root,
    load_base_and_merged_config,
    load_cache,
    make_main_config,
    materialize_full_cache_dataset,
    normalize_xy_array,
    prepare_experiment_dataset_env,
    resolve_ground_truth_path,
    run_cmd_to_log,
    save_json,
    snapshot_project_configs,
    summarize_segments,
    write_config,
)


def setup_workspace(analysis_config: dict[str, Any], run_id: str):
    project_root = get_project_root()
    project_cfg = analysis_config.get("project", {})
    dataset_cfg = analysis_config.get("dataset", {})
    base_config_path = str(project_cfg.get("base_config_path", "config.yaml"))
    experiments_root = str(project_cfg.get("experiments_root", "experiments"))
    base_config, merged_config = load_base_and_merged_config(project_root, base_config_path)
    use_symlink = bool(dataset_cfg.get("use_symlink", True))
    run_paths = create_run_workspace(project_root, experiments_root, run_id, use_symlink)
    snapshot_project_configs(project_root, run_paths, base_config_path, base_config)
    original_dataset_folder, experiment_dataset_folder = prepare_experiment_dataset_env(project_root, run_paths, base_config, use_symlink)
    save_json(
        {
            "run_id": run_id,
            "project_root": str(project_root),
            "run_dir": str(run_paths["run_dir"]),
            "base_config_path": str(project_root / base_config_path),
            "original_dataset_folder": str(original_dataset_folder),
            "experiment_dataset_folder": str(experiment_dataset_folder),
            "use_symlink": use_symlink,
        },
        run_paths["metadata_dir"] / "workspace_manifest.json",
    )
    return project_root, base_config, merged_config, run_paths, original_dataset_folder, experiment_dataset_folder, use_symlink


def run_training(
    *,
    project_root: Path,
    run_paths: dict[str, Path],
    base_config: dict[str, Any],
    original_dataset_folder: Path,
    experiment_dataset_folder: Path,
    use_symlink: bool,
    train_datasets: list[str],
    run_label: str,
    method: str,
    enable_neural_training: bool,
    enable_trajectory_decoding: bool,
    config_overrides: dict[str, Any],
    dry_run: bool = False,
) -> None:
    local_train_datasets = [
        materialize_full_cache_dataset(original_dataset_folder, experiment_dataset_folder, name, use_symlink)
        for name in train_datasets
    ]
    output_dir = f"output/{run_label}/training"
    checkpoint_dir = "checkpoint"
    training_output_dir = run_paths["run_dir"] / output_dir
    ensure_dir(training_output_dir)
    copy_neighbor_matrix_if_available(project_root, training_output_dir, use_symlink)
    cfg = make_main_config(
        base_config,
        csi_datasets=local_train_datasets,
        method=method,
        enable_neural_training=enable_neural_training,
        enable_trajectory_decoding=enable_trajectory_decoding,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        overrides=config_overrides,
    )
    config_path = write_config(cfg, run_paths, f"{run_label}_train.yaml")
    log_path = run_paths["logs_dir"] / run_label / "train.log"
    cmd = [sys.executable, str(project_root / "main.py"), "--config", str(config_path)]
    run_cmd_to_log(cmd, cwd=run_paths["run_dir"], log_path=log_path, dry_run=dry_run)


def run_segment_validation(
    *,
    project_root: Path,
    run_paths: dict[str, Path],
    base_config: dict[str, Any],
    merged_config: dict[str, Any],
    original_dataset_folder: Path,
    experiment_dataset_folder: Path,
    use_symlink: bool,
    target_dataset: str,
    target_gt_paths: dict[str, str],
    run_label: str,
    method: str,
    output_subdir: str,
    checkpoint_dir: str,
    enable_neural_training: bool,
    enable_trajectory_decoding: bool,
    config_overrides: dict[str, Any],
    segment_blocks: int,
    segment_stride: int,
    include_last_partial_segment: bool,
    dry_run: bool = False,
):
    """Validate by calling main.py once per segment."""
    materialize_full_cache_dataset(original_dataset_folder, experiment_dataset_folder, target_dataset, use_symlink)
    target_cache_path = get_dataset_cache_path(original_dataset_folder, target_dataset)
    target_cache = load_cache(target_cache_path)
    num_sample = int(config_overrides.get("NUM_SAMPLE", merged_config["NUM_SAMPLE"]))
    num_packet = int(config_overrides.get("NUM_PACKET", merged_config["NUM_PACKET"]))
    samples_per_block = num_sample * num_packet
    total_cache_samples = int(len(target_cache))
    total_blocks = total_cache_samples // samples_per_block
    leftover_samples = total_cache_samples % samples_per_block
    if total_blocks <= 0:
        raise RuntimeError(f"Target cache too short: {target_cache_path}")
    gt_path = resolve_ground_truth_path(original_dataset_folder, target_dataset, target_gt_paths, None, project_root)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    gt_raw = np.load(gt_path)
    gt_all_xy = normalize_xy_array(gt_raw)
    segment_plan = build_segment_plan(total_blocks, segment_blocks, segment_stride, include_last_partial_segment)
    save_json(
        {
            "run_label": run_label,
            "target_dataset": target_dataset,
            "target_cache_path": str(target_cache_path),
            "target_cache_shape": list(target_cache.shape),
            "target_gt_path": str(gt_path),
            "gt_shape": list(gt_raw.shape),
            "num_sample": num_sample,
            "num_packet": num_packet,
            "samples_per_block": samples_per_block,
            "total_cache_samples": total_cache_samples,
            "total_blocks": total_blocks,
            "leftover_samples": leftover_samples,
            "segment_blocks": segment_blocks,
            "segment_stride": segment_stride,
            "include_last_partial_segment": include_last_partial_segment,
            "main_py_call_scope": "one_segment",
            "segment_plan": segment_plan,
        },
        run_paths["metadata_dir"] / f"{run_label}_{target_dataset}_segment_plan.json",
    )
    rows = []
    all_errors = []
    for plan in segment_plan:
        seg_idx = int(plan["segment_index"])
        block_start = int(plan["block_start"])
        block_count = int(plan["block_count"])
        print(f"\n[{run_label} | {target_dataset} | Segment {seg_idx:04d}] block_start={block_start}, block_count={block_count}, samples={block_count * samples_per_block}")
        segment_dataset_name, segment_dataset_dir, segment_meta = create_segment_cache_dataset(
            experiment_dataset_folder, target_dataset, target_cache, samples_per_block, seg_idx, block_start, block_count
        )
        save_json(segment_meta, run_paths["metadata_dir"] / f"{run_label}_{target_dataset}_segment_{seg_idx:04d}.json")
        segment_output_dir = run_paths["output_root"] / output_subdir / target_dataset / f"seg_{seg_idx:04d}_b{block_start:06d}"
        ensure_dir(segment_output_dir)
        copy_neighbor_matrix_if_available(project_root, segment_output_dir, use_symlink)
        relative_output_dir = f"output/{output_subdir}/{target_dataset}/seg_{seg_idx:04d}_b{block_start:06d}"
        cfg = make_main_config(
            base_config,
            csi_datasets=[segment_dataset_name],
            method=method,
            enable_neural_training=enable_neural_training,
            enable_trajectory_decoding=enable_trajectory_decoding,
            output_dir=relative_output_dir,
            checkpoint_dir=checkpoint_dir,
            overrides=config_overrides,
        )
        config_path = write_config(cfg, run_paths, f"{run_label}_{target_dataset}_seg_{seg_idx:04d}_b{block_start:06d}.yaml")
        log_path = run_paths["logs_dir"] / run_label / target_dataset / f"seg_{seg_idx:04d}_b{block_start:06d}.log"
        cmd = [sys.executable, str(project_root / "main.py"), "--config", str(config_path)]
        run_cmd_to_log(cmd, cwd=run_paths["run_dir"], log_path=log_path, dry_run=dry_run)
        if dry_run:
            continue
        output_dataset_dir = segment_output_dir / segment_dataset_name
        row, errors, _pred_xy, gt_xy_segment = evaluate_segment_output(
            output_dataset_dir,
            gt_all_xy,
            target_dataset,
            run_label,
            seg_idx,
            block_start,
            block_count,
            num_sample,
            num_packet,
            samples_per_block,
            total_blocks,
            total_cache_samples,
        )
        row["segment_dataset_name"] = segment_dataset_name
        row["segment_dataset_dir"] = str(segment_dataset_dir)
        row["segment_sample_start"] = segment_meta["sample_start"]
        row["segment_sample_end"] = segment_meta["sample_end"]
        row["segment_protocol"] = "segment_cache_passed_to_main_py_once"
        np.save(segment_output_dir / "segment_ground_truth.npy", gt_xy_segment)
        np.save(segment_output_dir / "segment_errors.npy", errors)
        rows.append(row)
        all_errors.append(errors)
        print(f"[{run_label} | {target_dataset} | Segment {seg_idx:04d}] mean={row['mean_error']:.6f}, median={row['median_error']:.6f}, rmse={row['rmse_error']:.6f}, p90={row['p90_error']:.6f}")
    if dry_run:
        return None, rows
    summary = summarize_segments(rows, all_errors)
    summary.update(
        {
            "run_label": run_label,
            "target_dataset": target_dataset,
            "method": method,
            "enable_neural_training": bool(enable_neural_training),
            "enable_trajectory_decoding": bool(enable_trajectory_decoding),
            "config_overrides": json.dumps(config_overrides, ensure_ascii=False),
            "target_cache_path": str(target_cache_path),
            "target_gt_path": str(gt_path),
            "segment_blocks": segment_blocks,
            "segment_stride": segment_stride,
            "main_py_call_scope": "one_segment",
            "segment_error_protocol": "each segment is evaluated from one main.py output",
            "include_last_partial_segment": include_last_partial_segment,
            "num_sample": num_sample,
            "num_packet": num_packet,
            "samples_per_block": samples_per_block,
            "total_cache_samples": total_cache_samples,
            "total_blocks": total_blocks,
            "leftover_samples": leftover_samples,
            "all_outputs_under": str(run_paths["run_dir"]),
        }
    )
    return summary, rows
