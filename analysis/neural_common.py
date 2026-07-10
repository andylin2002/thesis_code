from __future__ import annotations

import json
from typing import Any

from common import (
    ensure_root_neighbor_matrix,
    flatten_for_csv,
    get_project_root,
    load_base_and_merged_config,
    save_json,
    write_csv,
)
from experiment_runner import run_segment_validation, run_training, setup_workspace


def ensure_neural_preflight(analysis_config: dict[str, Any], dry_run: bool = False) -> None:
    """Make sure project_root/output/neighbor_matrix.npy exists before any
    neural training/validation run starts.

    This must run once before the very first main.py invocation of a neural
    sweep (param search or seed search), because NeuralWorker needs this file
    the moment it starts and cannot generate it itself. See
    common.ensure_root_neighbor_matrix()'s docstring, and the "Known upstream
    caveats" note at the top of common.py, for the full explanation.

    This is a no-op (cheap existence check only) if the file already exists,
    so it is safe to call at the top of every sweep entry point.
    """
    project_root = get_project_root()
    project_cfg = analysis_config.get("project", {})
    dataset_cfg = analysis_config.get("dataset", {})
    neural_cfg = analysis_config.get("neural", {})

    base_config_path = str(project_cfg.get("base_config_path", "config.yaml"))
    experiments_root = str(project_cfg.get("experiments_root", "experiments"))
    default_method = str(project_cfg.get("default_method", "PROPOSED"))

    base_config, _ = load_base_and_merged_config(project_root, base_config_path)

    train_datasets = list(dataset_cfg.get("train_datasets", []))
    if not train_datasets:
        raise ValueError(
            "dataset.train_datasets must contain at least one dataset name; "
            "it is needed both for training and for the neighbor_matrix preflight."
        )

    preflight_cfg = neural_cfg.get("preflight", {}) or {}
    preflight_dataset = str(preflight_cfg.get("dataset") or train_datasets[0])
    fixed_common = neural_cfg.get("fixed_common", {}) or {}
    method = str(
        preflight_cfg.get("method")
        or fixed_common.get("METHOD")
        or base_config.get("METHOD")
        or default_method
    )

    ensure_root_neighbor_matrix(
        project_root=project_root,
        base_config=base_config,
        preflight_dataset=preflight_dataset,
        method=method,
        experiments_root=experiments_root,
        dry_run=dry_run,
    )


def build_neural_overrides(neural_cfg: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
    fixed_common = neural_cfg.get("fixed_common", {}) or {}
    if not isinstance(fixed_common, dict):
        raise TypeError("neural.fixed_common must be a dictionary.")
    overrides = dict(fixed_common)
    overrides.update(params)
    overrides["FIX_RANDOM_SEED"] = True
    overrides["RANDOM_SEED"] = int(seed)
    return overrides


def run_one_neural_experiment(analysis_config: dict[str, Any], run_id: str, run_label: str, seed: int, params: dict[str, Any], dry_run: bool = False) -> dict[str, Any] | None:
    project_cfg = analysis_config.get("project", {})
    dataset_cfg = analysis_config.get("dataset", {})
    eval_cfg = analysis_config.get("evaluation", {})
    neural_cfg = analysis_config.get("neural", {})
    project_root, base_config, merged_config, run_paths, original_dataset_folder, experiment_dataset_folder, use_symlink = setup_workspace(analysis_config, run_id)
    method = str(params.get("METHOD", neural_cfg.get("fixed_common", {}).get("METHOD", base_config.get("METHOD", project_cfg.get("default_method", "PROPOSED")))))
    train_datasets = list(dataset_cfg.get("train_datasets", ["train_long"]))
    target_datasets = list(dataset_cfg.get("target_datasets", ["train_wander"]))
    target_gt_paths = dataset_cfg.get("target_gt_paths", {}) or {}
    segment_blocks = int(eval_cfg.get("segment_blocks", 10))
    segment_stride = int(eval_cfg.get("segment_stride", 10))
    include_last_partial_segment = bool(eval_cfg.get("include_last_partial_segment", False))
    training_cfg = neural_cfg.get("training", {})
    validation_cfg = neural_cfg.get("validation", {})
    train_enable_neural = bool(training_cfg.get("enable_neural_training", True))
    train_enable_trajectory_decoding = bool(training_cfg.get("enable_trajectory_decoding", False))
    val_enable_neural = bool(validation_cfg.get("enable_neural_training", True))
    val_enable_trajectory_decoding = bool(validation_cfg.get("enable_trajectory_decoding", True))
    overrides = build_neural_overrides(neural_cfg, params, seed)
    save_json(
        {
            "run_id": run_id,
            "run_label": run_label,
            "seed": seed,
            "params": params,
            "overrides": overrides,
            "train_datasets": train_datasets,
            "target_datasets": target_datasets,
            "training": training_cfg,
            "validation": validation_cfg,
            "main_py_validation_call_scope": "one_segment",
            "dry_run": dry_run,
        },
        run_paths["metadata_dir"] / "neural_run_manifest.json",
    )
    run_training(
        project_root=project_root,
        run_paths=run_paths,
        base_config=base_config,
        original_dataset_folder=original_dataset_folder,
        experiment_dataset_folder=experiment_dataset_folder,
        use_symlink=use_symlink,
        train_datasets=train_datasets,
        run_label=run_label,
        method=method,
        enable_neural_training=train_enable_neural,
        enable_trajectory_decoding=train_enable_trajectory_decoding,
        config_overrides=overrides,
        dry_run=dry_run,
    )
    if dry_run:
        return None
    target_summaries = []
    all_rows = []
    for target_dataset in target_datasets:
        summary, rows = run_segment_validation(
            project_root=project_root,
            run_paths=run_paths,
            base_config=base_config,
            merged_config=merged_config,
            original_dataset_folder=original_dataset_folder,
            experiment_dataset_folder=experiment_dataset_folder,
            use_symlink=use_symlink,
            target_dataset=target_dataset,
            target_gt_paths=target_gt_paths,
            run_label=run_label,
            method=method,
            output_subdir=f"{run_label}/validation",
            checkpoint_dir="checkpoint",
            enable_neural_training=val_enable_neural,
            enable_trajectory_decoding=val_enable_trajectory_decoding,
            config_overrides=overrides,
            segment_blocks=segment_blocks,
            segment_stride=segment_stride,
            include_last_partial_segment=include_last_partial_segment,
            dry_run=False,
        )
        assert summary is not None
        summary.update({"run_id": run_id, "run_label": run_label, "seed": int(seed), "params": json.dumps(params, ensure_ascii=False), "target_dataset": target_dataset})
        metrics_dir = run_paths["metrics_dir"] / run_label / target_dataset
        write_csv([flatten_for_csv(r) for r in rows], metrics_dir / "segment_metrics.csv")
        save_json(rows, metrics_dir / "segment_metrics.json")
        save_json(summary, metrics_dir / "summary.json")
        target_summaries.append(summary)
        all_rows.extend(rows)
    selected = dict(target_summaries[0])
    selected["all_target_summaries"] = json.dumps(target_summaries, ensure_ascii=False)
    selected["run_dir"] = str(run_paths["run_dir"])
    out_dir = run_paths["metrics_dir"] / run_label
    write_csv([flatten_for_csv(s) for s in target_summaries], out_dir / "target_summaries.csv")
    write_csv([flatten_for_csv(r) for r in all_rows], out_dir / "all_segment_metrics.csv")
    save_json({"target_summaries": target_summaries}, out_dir / "target_summaries.json")
    return selected