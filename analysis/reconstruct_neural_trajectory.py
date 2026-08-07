from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


SEG_DIR_PATTERN = re.compile(r"^seg_(\d+)_b(\d+)$")
TRAJECTORY_FILENAME = "trajectory.npy"


def find_trajectory_file(segment_dir: Path) -> Path | None:
    """
    Find trajectory.npy inside one segment directory.

    Supports structures like:
      seg_0000_b000000/trajectory.npy

    and nested structures like:
      seg_0000_b000000/train_wander__seg_0000_b000000/trajectory.npy
    """
    direct = segment_dir / TRAJECTORY_FILENAME
    if direct.exists():
        return direct

    matches = sorted(segment_dir.glob(f"**/{TRAJECTORY_FILENAME}"))

    if not matches:
        return None

    if len(matches) > 1:
        print(f"[warn] Multiple trajectory.npy files found under {segment_dir}")
        for path in matches:
            print(f"       candidate: {path}")
        print(f"       using: {matches[0]}")

    return matches[0]


def collect_segment_dirs(root: Path) -> list[tuple[int, int, Path]]:
    """
    Collect segment directories directly under root.

    Expected:
      root/
        seg_0000_b000000/
        seg_0001_b000010/
        ...
    """
    segment_dirs = []

    for item in root.iterdir():
        if not item.is_dir():
            continue

        match = SEG_DIR_PATTERN.match(item.name)
        if match is None:
            continue

        seg_idx = int(match.group(1))
        block_start = int(match.group(2))
        segment_dirs.append((seg_idx, block_start, item))

    segment_dirs.sort(key=lambda x: x[1])
    return segment_dirs


def reconstruct(root: Path) -> np.ndarray:
    segment_dirs = collect_segment_dirs(root)

    if not segment_dirs:
        raise RuntimeError(
            f"No segment directories found under {root}. "
            "Expected folders like seg_0000_b000000."
        )

    trajectories = []
    used = []
    skipped = []

    for seg_idx, block_start, segment_dir in segment_dirs:
        traj_path = find_trajectory_file(segment_dir)

        if traj_path is None:
            skipped.append(segment_dir)
            print(f"[warn] No trajectory.npy found under {segment_dir}; skipped.")
            continue

        traj = np.load(traj_path)

        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(
                f"Invalid trajectory shape in {traj_path}: {traj.shape}. "
                "Expected shape [T, 2]."
            )

        trajectories.append(traj)
        used.append(
            {
                "seg_idx": seg_idx,
                "block_start": block_start,
                "length": traj.shape[0],
                "path": traj_path,
            }
        )

    if not trajectories:
        raise RuntimeError(f"No valid trajectory.npy files found under {root}.")

    print("\n[Segments used]")
    for item in used:
        print(
            f"  seg={item['seg_idx']:04d}, "
            f"block_start={item['block_start']:06d}, "
            f"length={item['length']}, "
            f"trajectory={item['path']}"
        )

    if skipped:
        print("\n[Skipped segments]")
        for path in skipped:
            print(f"  {path}")

    lengths = [t.shape[0] for t in trajectories]
    print(f"\n[Summary]")
    print(f"  number of used segments: {len(trajectories)}")
    print(f"  per-segment lengths: {sorted(set(lengths))}")

    full_trajectory = np.concatenate(trajectories, axis=0)

    print(f"  reconstructed shape: {full_trajectory.shape}")

    return full_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate trajectory.npy files from segmented neural validation output. "
            "The input root should directly contain seg_NNNN_bNNNNNN folders."
        )
    )

    parser.add_argument(
        "root",
        type=str,
        help=(
            "Folder directly containing seg_NNNN_bNNNNNN directories, e.g. "
            "experiments/.../validation/train_wander"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npy path. Default: <root>/reconstructed_trajectory.npy"
        ),
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else root / "reconstructed_trajectory.npy"
    )

    full_trajectory = reconstruct(root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, full_trajectory)

    print("\n[Done]")
    print(f"  saved: {output_path}")
    print(f"  shape: {full_trajectory.shape}")


if __name__ == "__main__":
    main()