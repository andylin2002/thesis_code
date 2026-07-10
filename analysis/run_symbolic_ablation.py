from __future__ import annotations

import argparse
import json
from typing import Any

from common import flatten_for_csv, get_project_root, load_yaml, save_json, timestamp_run_id, write_csv
from experiment_runner import run_segment_validation, setup_workspace


def resolve_case_names(requested_cases: list[str], case_defs: dict[str, Any]) -> list[str]:
    if not requested_cases:
        raise ValueError("No symbolic ablation cases requested.")
    if len(requested_cases) == 1 and requested_cases[0] == "all":
        return list(case_defs.keys())
    unknown = [name for name in requested_cases if name not in case_defs]
    if unknown:
        raise KeyError(f"Unknown symbolic ablation case(s): {unknown}. Available cases: {list(case_defs.keys())}")
    return requested_cases


def build_symbolic_overrides(case_name: str, case_def: dict[str, Any], controls: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
    soft_em_key = str(controls.get("soft_em_key", "ENABLE_SOFT_EM"))
    tof_gain_key = str(controls.get("tof_gain_key", "ENABLE_TOF_GAIN_WEIGHT"))
    multipath_key = str(controls.get("multipath_key", "ENABLE_MULTIPATH"))
    write_multipath = bool(controls.get("write_multipath_key_even_if_missing", True))
    overrides = {
        soft_em_key: bool(case_def.get("soft_em", True)),
        tof_gain_key: bool(case_def.get("tof_gain", True)),
    }
    if multipath_key in base_config or write_multipath:
        overrides[multipath_key] = bool(case_def.get("multipath", True))
    extra = case_def.get("extra_overrides", {})
    if extra:
        if not isinstance(extra, dict):
            raise TypeError(f"extra_overrides for case {case_name!r} must be a dictionary.")
        overrides.update(extra)
    return overrides


def run_symbolic_ablation(analysis_config: dict[str, Any], cases_override: list[str] | None = None, dry_run: bool = False) -> None:
    project_cfg = analysis_config.get("project", {})
    dataset_cfg = analysis_config.get("dataset", {})
    eval_cfg = analysis_config.get("evaluation", {})
    symbolic_cfg = analysis_config.get("symbolic", {})
    run_id = timestamp_run_id(str(symbolic_cfg.get("run_name_prefix", "symbolic_ablation")))
    project_root, base_config, merged_config, run_paths, original_dataset_folder, experiment_dataset_folder, use_symlink = setup_workspace(analysis_config, run_id)
    method = str(base_config.get("METHOD", project_cfg.get("default_method", "PROPOSED")))
    target_datasets = list(dataset_cfg.get("target_datasets", ["train_wander"]))
    target_gt_paths = dataset_cfg.get("target_gt_paths", {}) or {}
    segment_blocks = int(eval_cfg.get("segment_blocks", 10))
    segment_stride = int(eval_cfg.get("segment_stride", 10))
    include_last_partial_segment = bool(eval_cfg.get("include_last_partial_segment", False))
    case_defs = symbolic_cfg.get("cases", {})
    if not isinstance(case_defs, dict) or not case_defs:
        raise ValueError("symbolic.cases must be a non-empty dictionary.")
    requested_cases = cases_override if cases_override is not None else list(symbolic_cfg.get("run_cases", ["all"]))
    case_names = resolve_case_names(requested_cases, case_defs)
    controls = symbolic_cfg.get("controls", {})
    save_json(
        {
            "mode": "symbolic_ablation",
            "run_id": run_id,
            "run_dir": str(run_paths["run_dir"]),
            "method": method,
            "target_datasets": target_datasets,
            "case_names": case_names,
            "segment_blocks": segment_blocks,
            "segment_stride": segment_stride,
            "main_py_call_scope": "one_segment",
            "dry_run": dry_run,
        },
        run_paths["metadata_dir"] / "analysis_manifest.json",
    )
    all_summaries = []
    all_rows = []
    for case_name in case_names:
        case_def = case_defs[case_name]
        if not isinstance(case_def, dict):
            raise TypeError(f"Case {case_name!r} must be a dictionary.")
        overrides = build_symbolic_overrides(case_name, case_def, controls, base_config)
        print("\n" + "#" * 100)
        print(f"[Symbolic Case] {case_name}")
        print(f"Overrides: {overrides}")
        print("#" * 100)
        save_json({"case_name": case_name, "case_definition": case_def, "case_overrides": overrides}, run_paths["metadata_dir"] / f"symbolic_case_{case_name}.json")
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
                run_label=case_name,
                method=method,
                output_subdir=f"symbolic_ablation/{case_name}",
                checkpoint_dir=f"checkpoint/{case_name}",
                enable_neural_training=False,
                enable_trajectory_decoding=True,
                config_overrides=overrides,
                segment_blocks=segment_blocks,
                segment_stride=segment_stride,
                include_last_partial_segment=include_last_partial_segment,
                dry_run=dry_run,
            )
            if dry_run:
                continue
            assert summary is not None
            summary.update({"mode": "symbolic_ablation", "case_name": case_name, "target_dataset": target_dataset, "case_overrides": json.dumps(overrides, ensure_ascii=False)})
            metrics_dir = run_paths["metrics_dir"] / "symbolic_ablation" / case_name / target_dataset
            write_csv([flatten_for_csv(r) for r in rows], metrics_dir / "segment_metrics.csv")
            save_json(rows, metrics_dir / "segment_metrics.json")
            save_json(summary, metrics_dir / "summary.json")
            all_rows.extend(rows)
            all_summaries.append(summary)
    if not dry_run:
        out_dir = run_paths["metrics_dir"] / "symbolic_ablation"
        write_csv([flatten_for_csv(s) for s in all_summaries], out_dir / "all_cases_summary.csv")
        write_csv([flatten_for_csv(r) for r in all_rows], out_dir / "all_cases_segment_metrics.csv")
        save_json({"summaries": all_summaries}, out_dir / "all_cases_summary.json")
    print(f"\n[Done] Generated files are under:\n{run_paths['run_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", default="analysis/analysis_config.yaml")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    project_root = get_project_root()
    analysis_config = load_yaml(project_root / args.analysis_config)
    run_symbolic_ablation(analysis_config, cases_override=args.cases, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
