from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SEG_DIR_PATTERN = re.compile(r"^seg_(\d+)_b(\d+)$")
TRAJECTORY_FILENAME = "trajectory.npy"


def resolve_symbolic_output_root(experiment_root: Path) -> Path:
    """
    Accept either:
      experiments/symbolic_ablation_xxx
    or:
      experiments/symbolic_ablation_xxx/output/symbolic_ablation
    """
    if (experiment_root / "output" / "symbolic_ablation").is_dir():
        return experiment_root / "output" / "symbolic_ablation"

    if experiment_root.name == "symbolic_ablation":
        return experiment_root

    raise FileNotFoundError(
        "Cannot find symbolic ablation output folder. Expected either:\n"
        f"  {experiment_root}/output/symbolic_ablation\n"
        "or pass the output/symbolic_ablation folder directly."
    )


def find_trajectory_file(segment_dir: Path) -> Path | None:
    """
    Support nested segment output like:
      seg_0000_b000000/train_wander__seg_0000_b000000/trajectory.npy
    """
    direct = segment_dir / TRAJECTORY_FILENAME
    if direct.exists():
        return direct

    matches = sorted(segment_dir.glob(f"**/{TRAJECTORY_FILENAME}"))

    if not matches:
        return None

    if len(matches) > 1:
        print(f"[warn] Multiple trajectory.npy found under {segment_dir}")
        for m in matches:
            print(f"       candidate: {m}")
        print(f"       using: {matches[0]}")

    return matches[0]


def collect_segments(mode_dir: Path) -> list[tuple[int, int, Path]]:
    segments = []

    for item in mode_dir.iterdir():
        if not item.is_dir():
            continue

        m = SEG_DIR_PATTERN.match(item.name)
        if m is None:
            continue

        seg_idx = int(m.group(1))
        block_start = int(m.group(2))
        segments.append((seg_idx, block_start, item))

    segments.sort(key=lambda x: x[1])
    return segments


def reconstruct_one_mode(mode_dir: Path) -> tuple[np.ndarray, list[dict]]:
    segments = collect_segments(mode_dir)

    if not segments:
        raise RuntimeError(
            f"No segment folders found under {mode_dir}. "
            "Expected folders like seg_0000_b000000."
        )

    trajectories = []
    manifest = []

    for seg_idx, block_start, seg_dir in segments:
        traj_path = find_trajectory_file(seg_dir)

        if traj_path is None:
            print(f"[warn] No trajectory.npy found under {seg_dir}; skipped.")
            continue

        traj = np.load(traj_path)

        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(
                f"Invalid trajectory shape in {traj_path}: {traj.shape}. "
                "Expected [T, 2]."
            )

        trajectories.append(traj)
        manifest.append(
            {
                "seg_idx": seg_idx,
                "block_start": block_start,
                "segment_dir": str(seg_dir),
                "trajectory_path": str(traj_path),
                "length": int(traj.shape[0]),
            }
        )

    if not trajectories:
        raise RuntimeError(f"No valid trajectory.npy files found under {mode_dir}.")

    # IMPORTANT:
    # b000010 is block index, not timestamp index.
    # Therefore we do NOT infer overlap from b-stride.
    # For your setting, segments are non-overlapping:
    # 10 blocks per segment × NUM_SAMPLE=15 = 150 timestamps.
    full_traj = np.concatenate(trajectories, axis=0)

    return full_traj, manifest


def find_case_mode_dirs(symbolic_root: Path, dataset_name: str | None = None) -> list[tuple[str, str, Path]]:
    """
    Find:
      output/symbolic_ablation/ablation_000/train_wander
      output/symbolic_ablation/ablation_001/train_wander
      ...
    """
    found = []

    for case_dir in sorted(symbolic_root.iterdir()):
        if not case_dir.is_dir():
            continue

        if case_dir.name == "reconstructed_trajectories":
            continue

        # Usually: ablation_000, ablation_001, ...
        # But this also supports full, no_softem, etc.
        for mode_dir in sorted(case_dir.iterdir()):
            if not mode_dir.is_dir():
                continue

            if dataset_name is not None and mode_dir.name != dataset_name:
                continue

            if collect_segments(mode_dir):
                found.append((case_dir.name, mode_dir.name, mode_dir))

    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct full trajectories for every symbolic ablation case. "
            "Input can be the experiment root, e.g. experiments/symbolic_ablation_YYYYMMDD_HHMMSS."
        )
    )

    parser.add_argument(
        "experiment_root",
        type=str,
        help=(
            "Experiment root, e.g. experiments/symbolic_ablation_20260710_113839, "
            "or the output/symbolic_ablation folder directly."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only reconstruct this dataset/mode, e.g. train_wander. Default: all datasets found.",
    )

    parser.add_argument(
        "--save-inside-case",
        action="store_true",
        default=True,
        help=(
            "Save reconstructed_trajectory.npy inside each case/dataset folder. "
            "Default: enabled."
        ),
    )

    parser.add_argument(
        "--central-output-dir",
        type=str,
        default=None,
        help=(
            "Optional central output directory. If omitted, uses "
            "<output/symbolic_ablation>/reconstructed_trajectories."
        ),
    )

    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).resolve()

    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {experiment_root}")

    symbolic_root = resolve_symbolic_output_root(experiment_root)

    if args.central_output_dir is not None:
        central_output_dir = Path(args.central_output_dir).resolve()
    else:
        central_output_dir = symbolic_root / "reconstructed_trajectories"

    central_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Root]")
    print(f"  experiment root: {experiment_root}")
    print(f"  symbolic output: {symbolic_root}")
    print(f"  central output:  {central_output_dir}")
    print()

    case_mode_dirs = find_case_mode_dirs(symbolic_root, dataset_name=args.dataset)

    if not case_mode_dirs:
        raise RuntimeError(
            f"No case/dataset folders containing seg_NNNN_bNNNNNN were found under {symbolic_root}."
        )

    all_manifest = []

    for case_name, dataset_name, mode_dir in case_mode_dirs:
        print("=" * 100)
        print(f"[Reconstruct] case={case_name}, dataset={dataset_name}")
        print(f"  mode dir: {mode_dir}")

        full_traj, manifest = reconstruct_one_mode(mode_dir)

        print(f"  segments used: {len(manifest)}")
        print(f"  per-segment lengths: {sorted(set(m['length'] for m in manifest))}")
        print(f"  reconstructed shape: {full_traj.shape}")

        # Save inside each case/dataset folder
        if args.save_inside_case:
            inside_path = mode_dir / "reconstructed_trajectory.npy"
            np.save(inside_path, full_traj)
            print(f"  saved inside case: {inside_path}")

        # Save central copy
        central_name = f"{case_name}__{dataset_name}__reconstructed_trajectory.npy"
        central_path = central_output_dir / central_name
        np.save(central_path, full_traj)
        print(f"  saved central copy: {central_path}")

        case_manifest = {
            "case_name": case_name,
            "dataset_name": dataset_name,
            "mode_dir": str(mode_dir),
            "reconstructed_shape": list(full_traj.shape),
            "central_output_path": str(central_path),
            "inside_case_output_path": str(mode_dir / "reconstructed_trajectory.npy"),
            "segments": manifest,
        }

        manifest_path = central_output_dir / f"{case_name}__{dataset_name}__manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(case_manifest, f, indent=2, ensure_ascii=False)

        all_manifest.append(case_manifest)
        print()

    all_manifest_path = central_output_dir / "all_reconstruction_manifest.json"
    with open(all_manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("[Done]")
    print(f"  reconstructed cases: {len(all_manifest)}")
    print(f"  manifest: {all_manifest_path}")


if __name__ == "__main__":
    main()