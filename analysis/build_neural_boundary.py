from __future__ import annotations

"""
build_neural_boundary.py

Place this file in <project_root>/analysis/ and run it from the project root.

Purpose
-------
Build experiments/neural_boundary/ from two existing experiment trees:

1. A complete symbolic-only run, used as the canonical source of
   emission_log_probs_qgt.npy, segment ground truth, grid, and neighbor matrix.
2. A neural-enabled run, used only as the source of reliability.npy.

Segments from the two trees are matched by the global block offset embedded in
folder names such as "..._b000020". The script then processes every block
independently and saves:

    reliability_neural_concat.npy
    reliability_emission_oracle_concat.npy
    reliability_decoding_oracle_concat.npy

and the four block-wise decoded trajectories:

    trajectory_uniform_concat.npy
    trajectory_neural_concat.npy
    trajectory_emission_oracle_concat.npy
    trajectory_decoding_oracle_concat.npy

The emission oracle follows the supplied Chapter 8 implementation:

    z_AP(log support at GT)
    + 0.3 * z_AP(margin against the best competing grid)
    + 0.05 * z_AP(negative spatial entropy)

The decoding oracle searches one AP-bias vector per block. The vector is fixed
across all T samples in that block. Each candidate is accepted only when it
reduces that block's mean localization error after Viterbi decoding.

Important
---------
Edit SYMBOLIC_RUN_DIR and NEURAL_RUN_DIR below, or pass both paths on the
command line. They may point either to the experiment output root or directly
to a subtree containing the merged segment folders.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


# =============================================================================
# User-editable defaults
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

# Complete symbolic-only ablation (SoftEM=1, ToF/Gain=1, Multipath=1, Neural=0).
SYMBOLIC_RUN_DIR = (
    EXPERIMENTS_ROOT
    / "symbolic_ablation_20260710_113839"
    / "output"
    / "symbolic_ablation"
    / "ablation_111"
    / "train_wander"
)

# Change this to the selected neural seed-search/final-evaluation output tree.
# The folder may be, for example:
#   experiments/<experiment>/runs/seed_XXXX_rep_00/output/validation_segments
NEURAL_RUN_DIR = (
    EXPERIMENTS_ROOT
    / "neural_seed_search_20260711_092809"
    / "runs"
    / "seed_0018_rep_00"
    / "output"
    / "seed_0018_rep_00"
    / "validation"
    / "train_wander"
)

OUTPUT_DIR = EXPERIMENTS_ROOT / "neural_boundary"

NUM_SAMPLE = 15
DEVICE = "cpu"
TEMPERATURE = 1.0
MARGIN_WEIGHT = 0.3
CONCENTRATION_WEIGHT = 0.05
SEARCH_STEPS = (2.0, 1.0, 0.5, 0.25)
MAX_PASSES_PER_STEP = 2
DECODING_ORACLE_INIT = "uniform"  # "uniform" or "emission"


# =============================================================================
# Generic helpers
# =============================================================================

_BLOCK_OFFSET_RE = re.compile(r"_b(\d+)(?:\D|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build neural, emission-oracle, and decoding-oracle reliability arrays."
    )
    parser.add_argument("--symbolic-run-dir", type=Path, default=SYMBOLIC_RUN_DIR)
    parser.add_argument("--neural-run-dir", type=Path, default=NEURAL_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-sample", type=int, default=NUM_SAMPLE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--margin-weight", type=float, default=MARGIN_WEIGHT)
    parser.add_argument("--concentration-weight", type=float, default=CONCENTRATION_WEIGHT)
    parser.add_argument(
        "--search-steps",
        type=str,
        default=",".join(str(x) for x in SEARCH_STEPS),
        help="Comma-separated decoding-oracle coordinate-search steps.",
    )
    parser.add_argument("--max-passes-per-step", type=int, default=MAX_PASSES_PER_STEP)
    parser.add_argument(
        "--decoding-oracle-init",
        choices=["uniform", "emission"],
        default=DECODING_ORACLE_INIT,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_search_steps(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(x <= 0 for x in values):
        raise ValueError(f"Invalid --search-steps: {text!r}")
    return values


def parse_block_offset(path: Path) -> int:
    """Find the last b###### token in the path, preferring the closest folder."""
    for part in reversed(path.parts):
        matches = list(_BLOCK_OFFSET_RE.finditer(part))
        if matches:
            return int(matches[-1].group(1))
    raise ValueError(f"Cannot parse a global block offset from: {path}")


def normalize_xy(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr.reshape(-1, arr.shape[-1])[:, :2].astype(np.float32)
    raise ValueError(f"Unsupported coordinate shape: {arr.shape}")


def reliability_to_qt(rel: np.ndarray, q: int, t_total: int) -> np.ndarray:
    """Convert [Q,T], [T,Q], [B,Q,T], or [B,T,Q] to [Q,T]."""
    rel = np.asarray(rel, dtype=np.float32)

    if rel.ndim == 2:
        if rel.shape == (q, t_total):
            return rel.copy()
        if rel.shape == (t_total, q):
            return rel.T.copy()

    if rel.ndim == 3:
        b, d1, d2 = rel.shape
        if d1 == q and b * d2 == t_total:
            return rel.transpose(1, 0, 2).reshape(q, t_total).copy()
        if d2 == q and b * d1 == t_total:
            return rel.transpose(2, 0, 1).reshape(q, t_total).copy()

    raise ValueError(
        f"Cannot convert reliability shape {rel.shape} to [Q,T]=[{q},{t_total}]."
    )


def normalize_unit_mean_per_t(r_qt: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Enforce mean_q r[q,t] = 1, equivalently sum_q r[q,t] = Q."""
    r = np.asarray(r_qt, dtype=np.float32)
    if not np.all(np.isfinite(r)):
        raise ValueError("Reliability contains NaN or infinity.")
    if np.any(r < 0):
        raise ValueError(f"Reliability contains negative values; min={float(r.min())}")
    mean_t = r.mean(axis=0, keepdims=True)
    if np.any(mean_t <= eps):
        bad = np.flatnonzero(mean_t.reshape(-1) <= eps)
        raise ValueError(f"Reliability has zero-mean columns at time indices {bad[:10].tolist()}")
    return (r / mean_t).astype(np.float32)


def find_nearest_file(start_dir: Path, filename: str, search_root: Path) -> Path:
    """Look in start_dir and its ancestors, then in the nearby segment subtree."""
    start_dir = start_dir.resolve()
    search_root = search_root.resolve()

    current = start_dir
    while True:
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current == search_root or current.parent == current:
            break
        current = current.parent

    # Some experiment layouts put grid.npy in the segment folder but the merged
    # dataset one directory below it. The ancestor check above covers that. This
    # final search supports slightly different layouts while rejecting ambiguity.
    local_candidates = list(start_dir.parent.glob(f"**/{filename}"))
    local_candidates = [p for p in local_candidates if p.is_file()]
    if len(local_candidates) == 1:
        return local_candidates[0]

    raise FileNotFoundError(
        f"Could not find an unambiguous {filename} for segment {start_dir}. "
        f"Checked ancestors up to {search_root}."
    )


def discover_symbolic_segments(root: Path) -> dict[int, Path]:
    """Find merged symbolic folders containing emission with segment GT nearby.

    In the current experiment layout, emission_log_probs_qgt.npy is stored in
    the inner merged-dataset folder, while segment_ground_truth.npy is stored
    in its parent seg_XXXX_bYYYYYY folder. Therefore GT must be searched in
    the emission folder and its ancestors rather than required in the same
    directory.
    """
    found: dict[int, list[Path]] = {}
    for emission_path in root.rglob("emission_log_probs_qgt.npy"):
        folder = emission_path.parent
        try:
            find_nearest_file(folder, "segment_ground_truth.npy", root)
            offset = parse_block_offset(folder)
        except (FileNotFoundError, ValueError):
            continue
        found.setdefault(offset, []).append(folder)

    if not found:
        raise FileNotFoundError(
            f"No merged symbolic segment containing emission_log_probs_qgt.npy and "
            f"segment_ground_truth.npy was found under {root}."
        )

    result: dict[int, Path] = {}
    for offset, candidates in found.items():
        # Prefer a folder carrying merged_segment_meta.json. Then prefer the
        # candidate with the longest emission time axis, which rejects one-block
        # artifacts if both merged and temporary outputs are present.
        ranked: list[tuple[int, int, Path]] = []
        for folder in candidates:
            shape = np.load(folder / "emission_log_probs_qgt.npy", mmap_mode="r").shape
            if len(shape) != 3:
                continue
            merged_flag = int((folder / "merged_segment_meta.json").exists())
            ranked.append((merged_flag, int(shape[2]), folder))
        if not ranked:
            raise ValueError(f"No valid [Q,G,T] symbolic emission for block offset {offset}.")
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = ranked[0]
        tied = [x for x in ranked if x[:2] == best[:2]]
        if len(tied) > 1:
            raise RuntimeError(
                f"Ambiguous symbolic folders for block offset {offset}: "
                + ", ".join(str(x[2]) for x in tied)
            )
        result[offset] = best[2]

    return dict(sorted(result.items()))


def discover_neural_reliabilities(root: Path) -> dict[int, list[Path]]:
    found: dict[int, list[Path]] = {}
    for rel_path in root.rglob("reliability.npy"):
        try:
            offset = parse_block_offset(rel_path.parent)
        except ValueError:
            continue
        found.setdefault(offset, []).append(rel_path)
    if not found:
        raise FileNotFoundError(f"No reliability.npy was found under {root}.")
    return found


def choose_neural_reliability(
    candidates: list[Path], *, q: int, t_total: int
) -> tuple[Path, np.ndarray]:
    valid: list[tuple[int, Path, np.ndarray]] = []
    errors: list[str] = []

    for path in candidates:
        try:
            rel_qt = reliability_to_qt(np.load(path), q=q, t_total=t_total)
            merged_flag = int(
                (path.parent / "merged_segment_meta.json").exists()
                or (path.parent / "segment_ground_truth.npy").exists()
            )
            valid.append((merged_flag, path, rel_qt))
        except Exception as exc:  # keep diagnostics for all candidates
            errors.append(f"{path}: {exc}")

    if not valid:
        details = "\n".join(errors)
        raise ValueError(
            f"No neural reliability candidate matches [Q,T]=[{q},{t_total}].\n{details}"
        )

    valid.sort(key=lambda x: x[0], reverse=True)
    best_flag = valid[0][0]
    best = [x for x in valid if x[0] == best_flag]
    if len(best) > 1:
        raise RuntimeError(
            "Multiple neural reliability files match the same block offset and shape:\n"
            + "\n".join(str(x[1]) for x in best)
        )

    _, path, rel_qt = best[0]
    return path, normalize_unit_mean_per_t(rel_qt)


# =============================================================================
# Project decoder
# =============================================================================


def import_decoder_objects():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from engines.symbolic_engine.stages.result_estimation._common.viterbi import (
        Viterbi_Algorithm,
    )
    from engines.symbolic_engine.stages.result_estimation.proposed import soft_em_utils

    return Viterbi_Algorithm, soft_em_utils


def aggregate_emission(
    emission_qgt: torch.Tensor, reliability_qt: torch.Tensor
) -> torch.Tensor:
    """Apply the same AP contribution rule used by the supplied implementation."""
    if emission_qgt.ndim != 3:
        raise ValueError(f"Expected emission [Q,G,T], got {tuple(emission_qgt.shape)}")
    q, _, t = emission_qgt.shape
    if tuple(reliability_qt.shape) != (q, t):
        raise ValueError(
            f"Reliability shape {tuple(reliability_qt.shape)} does not match {(q, t)}."
        )
    return (emission_qgt * reliability_qt.unsqueeze(1)).sum(dim=0)


def run_viterbi(
    emission_gt: torch.Tensor,
    grid: torch.Tensor,
    neighbor: torch.Tensor,
    device: torch.device,
    Viterbi_Algorithm,
    soft_em_utils,
) -> torch.Tensor:
    g, t = emission_gt.shape
    viterbi = Viterbi_Algorithm(g, t, grid, device)
    trajectory, _ = viterbi.run(
        emission_log_probs=emission_gt,
        neighbor_index_matrix=neighbor,
        get_max_previous_score=soft_em_utils.get_max_previous_score,
        transition_log_probs=None,
    )
    return trajectory


def decode_one_block(
    emission_qgt: torch.Tensor,
    reliability_qt: torch.Tensor,
    grid: torch.Tensor,
    neighbor: torch.Tensor,
    device: torch.device,
    Viterbi_Algorithm,
    soft_em_utils,
) -> np.ndarray:
    aggregated = aggregate_emission(emission_qgt, reliability_qt)
    trajectory = run_viterbi(
        aggregated,
        grid,
        neighbor,
        device,
        Viterbi_Algorithm,
        soft_em_utils,
    )
    return trajectory.detach().cpu().numpy().astype(np.float32)


def point_errors(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    if pred_xy.shape != gt_xy.shape:
        raise ValueError(f"Trajectory/GT shape mismatch: {pred_xy.shape} vs {gt_xy.shape}")
    return np.linalg.norm(
        pred_xy.astype(np.float64) - gt_xy.astype(np.float64), axis=1
    )


# =============================================================================
# Emission oracle
# =============================================================================


def zscore_across_aps(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def coords_to_grid_index(gt_xy: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    diff = gt_xy[:, None, :] - grid_xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return np.argmin(dist2, axis=1).astype(np.int64)


def compute_emission_oracle_score(
    emission_qgt: torch.Tensor,
    gt_idx_t: torch.Tensor,
    margin_weight: float,
    concentration_weight: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute [Q,T] emission-oracle scores for one block."""
    _, _, t = emission_qgt.shape
    time_idx = torch.arange(t, device=emission_qgt.device)

    log_spatial_prob = torch.log_softmax(emission_qgt, dim=1)
    spatial_prob = torch.softmax(emission_qgt, dim=1)

    support = log_spatial_prob[:, gt_idx_t, time_idx]

    masked = emission_qgt.clone()
    masked[:, gt_idx_t, time_idx] = -1e9
    best_other = masked.max(dim=1).values
    margin = support - best_other

    entropy = -(
        spatial_prob * torch.log(spatial_prob.clamp_min(eps))
    ).sum(dim=1)
    concentration = -entropy

    return (
        zscore_across_aps(support, eps=eps)
        + margin_weight * zscore_across_aps(margin, eps=eps)
        + concentration_weight * zscore_across_aps(concentration, eps=eps)
    )


def score_to_reliability(score_qt: torch.Tensor, temperature: float) -> torch.Tensor:
    q = score_qt.shape[0]
    return q * torch.softmax(score_qt / temperature, dim=0)


# =============================================================================
# Decoding oracle
# =============================================================================


def initial_decoding_score(
    emission_score_qt: torch.Tensor, init_mode: str
) -> torch.Tensor:
    q, t = emission_score_qt.shape
    if init_mode == "uniform":
        score_q = torch.zeros(q, device=emission_score_qt.device, dtype=emission_score_qt.dtype)
    elif init_mode == "emission":
        score_q = emission_score_qt.mean(dim=1)
        score_q = score_q - score_q.mean()
    else:
        raise ValueError(f"Unknown decoding oracle initialization: {init_mode}")
    return score_q[:, None].repeat(1, t)


def optimize_decoding_oracle_for_block(
    *,
    emission_block: torch.Tensor,
    gt_block_xy: np.ndarray,
    emission_score_block: torch.Tensor,
    grid: torch.Tensor,
    neighbor: torch.Tensor,
    temperature: float,
    search_steps: list[float],
    max_passes_per_step: int,
    init_mode: str,
    device: torch.device,
    Viterbi_Algorithm,
    soft_em_utils,
    block_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Search one AP bias vector that is constant throughout this block."""
    q, _, t = emission_block.shape
    init_score_qt = initial_decoding_score(emission_score_block, init_mode)
    bias_q = torch.zeros(q, device=device, dtype=emission_block.dtype)

    def make_reliability(candidate_bias: torch.Tensor) -> torch.Tensor:
        centered = candidate_bias - candidate_bias.mean()
        score_qt = init_score_qt + centered[:, None]
        return score_to_reliability(score_qt, temperature)

    def evaluate(candidate_bias: torch.Tensor) -> tuple[float, torch.Tensor, np.ndarray, np.ndarray]:
        rel = make_reliability(candidate_bias)
        traj = decode_one_block(
            emission_block,
            rel,
            grid,
            neighbor,
            device,
            Viterbi_Algorithm,
            soft_em_utils,
        )
        errors = point_errors(traj, gt_block_xy)
        return float(errors.mean()), rel, traj, errors

    best_error, best_rel, best_traj, best_errors = evaluate(bias_q)
    history: list[dict[str, Any]] = [
        {"stage": "init", "mean_error": best_error}
    ]
    accepted: list[dict[str, Any]] = []
    total_evaluations = 1

    print(f"[DecodingOracle:{block_name}] init mean={best_error:.6f}")

    for step in search_steps:
        for pass_idx in range(max_passes_per_step):
            pass_start = best_error
            accepted_count = 0

            for qi in range(q):
                local_best_error = best_error
                local_best_bias: Optional[torch.Tensor] = None
                local_best_rel: Optional[torch.Tensor] = None
                local_best_traj: Optional[np.ndarray] = None
                local_best_errors: Optional[np.ndarray] = None
                local_best_sign: Optional[float] = None

                for sign in (+1.0, -1.0):
                    trial_bias = bias_q.clone()
                    trial_bias[qi] += sign * step
                    trial_bias -= trial_bias.mean()

                    trial_error, trial_rel, trial_traj, trial_errors = evaluate(trial_bias)
                    total_evaluations += 1

                    if trial_error + 1e-10 < local_best_error:
                        local_best_error = trial_error
                        local_best_bias = trial_bias
                        local_best_rel = trial_rel
                        local_best_traj = trial_traj
                        local_best_errors = trial_errors
                        local_best_sign = sign

                if local_best_bias is not None and local_best_error + 1e-10 < best_error:
                    previous = best_error
                    bias_q = local_best_bias
                    best_error = local_best_error
                    best_rel = local_best_rel  # type: ignore[assignment]
                    best_traj = local_best_traj  # type: ignore[assignment]
                    best_errors = local_best_errors  # type: ignore[assignment]
                    accepted_count += 1
                    accepted.append(
                        {
                            "step": float(step),
                            "pass": int(pass_idx),
                            "ap": int(qi),
                            "sign": float(local_best_sign),
                            "previous_mean_error": float(previous),
                            "new_mean_error": float(best_error),
                        }
                    )

            history.append(
                {
                    "stage": f"step_{step}_pass_{pass_idx}",
                    "start_mean_error": float(pass_start),
                    "end_mean_error": float(best_error),
                    "accepted": int(accepted_count),
                    "total_evaluations": int(total_evaluations),
                }
            )
            print(
                f"[DecodingOracle:{block_name}] step={step:g} pass={pass_idx} "
                f"start={pass_start:.6f} end={best_error:.6f} accepted={accepted_count}"
            )

            if accepted_count == 0:
                break

    assert best_rel is not None and best_traj is not None and best_errors is not None
    return (
        best_rel.detach().cpu().numpy().astype(np.float32),
        best_traj.astype(np.float32),
        best_errors.astype(np.float64),
        {
            "block_name": block_name,
            "search_scope": "single_block",
            "constant_ap_vector_within_block": True,
            "initialization": init_mode,
            "final_bias_q": bias_q.detach().cpu().numpy().astype(float).tolist(),
            "initial_mean_error": float(history[0]["mean_error"]),
            "final_mean_error": float(best_error),
            "total_evaluations": int(total_evaluations),
            "history": history,
            "accepted_trials": accepted,
        },
    )


# =============================================================================
# Segment processing
# =============================================================================


def process_segment(
    *,
    offset: int,
    symbolic_dir: Path,
    neural_candidates: list[Path],
    symbolic_root: Path,
    num_sample: int,
    device: torch.device,
    temperature: float,
    margin_weight: float,
    concentration_weight: float,
    search_steps: list[float],
    max_passes_per_step: int,
    decoding_oracle_init: str,
    Viterbi_Algorithm,
    soft_em_utils,
) -> dict[str, Any]:
    emission_path = symbolic_dir / "emission_log_probs_qgt.npy"
    gt_path = find_nearest_file(symbolic_dir, "segment_ground_truth.npy", symbolic_root)
    trajectory_path = symbolic_dir / "trajectory.npy"

    emission_np = np.load(emission_path).astype(np.float32)
    if emission_np.ndim != 3:
        raise ValueError(f"Expected [Q,G,T] emission at {emission_path}, got {emission_np.shape}")
    q, g, t_total = emission_np.shape

    if t_total % num_sample != 0:
        raise ValueError(
            f"Segment at block offset {offset} has T={t_total}, not divisible by block length {num_sample}."
        )

    gt_xy = normalize_xy(np.load(gt_path))
    if gt_xy.shape != (t_total, 2):
        raise ValueError(
            f"GT shape mismatch at block offset {offset}: expected {(t_total, 2)}, got {gt_xy.shape}."
        )

    grid_path = find_nearest_file(symbolic_dir, "grid.npy", symbolic_root)
    neighbor_path = find_nearest_file(symbolic_dir, "neighbor_matrix.npy", symbolic_root)
    grid_np = np.load(grid_path).astype(np.float32)
    neighbor_np = np.load(neighbor_path).astype(np.int64)

    if grid_np.shape != (g, 2):
        raise ValueError(f"Grid shape {grid_np.shape} does not match emission G={g}: {grid_path}")
    if neighbor_np.ndim != 2 or neighbor_np.shape[0] != g:
        raise ValueError(
            f"Neighbor shape {neighbor_np.shape} does not match emission G={g}: {neighbor_path}"
        )

    neural_path, neural_qt_np = choose_neural_reliability(
        neural_candidates, q=q, t_total=t_total
    )

    emission = torch.from_numpy(emission_np).to(device=device, dtype=torch.float32)
    grid = torch.from_numpy(grid_np).to(device=device, dtype=torch.float32)
    neighbor = torch.from_numpy(neighbor_np).to(device=device, dtype=torch.long)
    neural_qt = torch.from_numpy(neural_qt_np).to(device=device, dtype=torch.float32)

    gt_idx_np = coords_to_grid_index(gt_xy, grid_np)

    rel_uniform_parts: list[np.ndarray] = []
    rel_neural_parts: list[np.ndarray] = []
    rel_emission_parts: list[np.ndarray] = []
    rel_decoding_parts: list[np.ndarray] = []

    traj_uniform_parts: list[np.ndarray] = []
    traj_neural_parts: list[np.ndarray] = []
    traj_emission_parts: list[np.ndarray] = []
    traj_decoding_parts: list[np.ndarray] = []

    decoding_meta: list[dict[str, Any]] = []

    num_blocks = t_total // num_sample
    for block_in_segment in range(num_blocks):
        t0 = block_in_segment * num_sample
        t1 = t0 + num_sample

        emission_b = emission[:, :, t0:t1]
        gt_b = gt_xy[t0:t1]
        gt_idx_b = torch.from_numpy(gt_idx_np[t0:t1]).to(device=device, dtype=torch.long)

        uniform_b = torch.ones((q, num_sample), device=device, dtype=torch.float32)
        neural_b = neural_qt[:, t0:t1]

        emission_score_b = compute_emission_oracle_score(
            emission_b,
            gt_idx_b,
            margin_weight=margin_weight,
            concentration_weight=concentration_weight,
        )
        emission_rel_b = score_to_reliability(emission_score_b, temperature)

        uniform_traj_b = decode_one_block(
            emission_b, uniform_b, grid, neighbor, device, Viterbi_Algorithm, soft_em_utils
        )
        neural_traj_b = decode_one_block(
            emission_b, neural_b, grid, neighbor, device, Viterbi_Algorithm, soft_em_utils
        )
        emission_traj_b = decode_one_block(
            emission_b, emission_rel_b, grid, neighbor, device, Viterbi_Algorithm, soft_em_utils
        )

        global_block = offset + block_in_segment
        decoding_rel_b_np, decoding_traj_b, _, block_search_meta = (
            optimize_decoding_oracle_for_block(
                emission_block=emission_b,
                gt_block_xy=gt_b,
                emission_score_block=emission_score_b,
                grid=grid,
                neighbor=neighbor,
                temperature=temperature,
                search_steps=search_steps,
                max_passes_per_step=max_passes_per_step,
                init_mode=decoding_oracle_init,
                device=device,
                Viterbi_Algorithm=Viterbi_Algorithm,
                soft_em_utils=soft_em_utils,
                block_name=f"b{global_block:06d}",
            )
        )

        rel_uniform_parts.append(uniform_b.detach().cpu().numpy().astype(np.float32))
        rel_neural_parts.append(neural_b.detach().cpu().numpy().astype(np.float32))
        rel_emission_parts.append(emission_rel_b.detach().cpu().numpy().astype(np.float32))
        rel_decoding_parts.append(decoding_rel_b_np)

        traj_uniform_parts.append(uniform_traj_b)
        traj_neural_parts.append(neural_traj_b)
        traj_emission_parts.append(emission_traj_b)
        traj_decoding_parts.append(decoding_traj_b)
        decoding_meta.append(block_search_meta)

    arrays = {
        "ground_truth": gt_xy.astype(np.float32),
        "gt_grid_idx": gt_idx_np.astype(np.int64),
        "reliability_uniform": np.concatenate(rel_uniform_parts, axis=1),
        "reliability_neural": np.concatenate(rel_neural_parts, axis=1),
        "reliability_emission_oracle": np.concatenate(rel_emission_parts, axis=1),
        "reliability_decoding_oracle": np.concatenate(rel_decoding_parts, axis=1),
        "trajectory_uniform": np.concatenate(traj_uniform_parts, axis=0),
        "trajectory_neural": np.concatenate(traj_neural_parts, axis=0),
        "trajectory_emission_oracle": np.concatenate(traj_emission_parts, axis=0),
        "trajectory_decoding_oracle": np.concatenate(traj_decoding_parts, axis=0),
    }

    # Optional sanity comparison with the saved symbolic trajectory.
    saved_uniform_delta: Optional[dict[str, float]] = None
    if trajectory_path.exists():
        saved_uniform = normalize_xy(np.load(trajectory_path))
        if saved_uniform.shape == arrays["trajectory_uniform"].shape:
            delta = point_errors(saved_uniform, arrays["trajectory_uniform"])
            saved_uniform_delta = {
                "mean": float(delta.mean()),
                "max": float(delta.max()),
            }

    meta = {
        "block_offset": int(offset),
        "num_blocks": int(num_blocks),
        "num_sample": int(num_sample),
        "q": int(q),
        "g": int(g),
        "t_total": int(t_total),
        "symbolic_dir": str(symbolic_dir),
        "emission_path": str(emission_path),
        "ground_truth_path": str(gt_path),
        "neural_reliability_path": str(neural_path),
        "grid_path": str(grid_path),
        "neighbor_path": str(neighbor_path),
        "saved_uniform_vs_redecoded_delta": saved_uniform_delta,
        "decoding_oracle_blocks": decoding_meta,
    }

    return {"arrays": arrays, "meta": meta, "grid": grid_np, "neighbor": neighbor_np}


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    symbolic_root = resolve_path(args.symbolic_run_dir)
    neural_root = resolve_path(args.neural_run_dir)
    output_dir = resolve_path(args.output_dir)

    if not symbolic_root.exists():
        raise FileNotFoundError(f"Symbolic run directory not found: {symbolic_root}")
    if not neural_root.exists():
        raise FileNotFoundError(f"Neural run directory not found: {neural_root}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite after confirming that it is safe to replace its generated files."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_out = output_dir / "segments"
    segments_out.mkdir(parents=True, exist_ok=True)

    search_steps = parse_search_steps(args.search_steps)
    device = torch.device(args.device)
    Viterbi_Algorithm, soft_em_utils = import_decoder_objects()

    symbolic_segments = discover_symbolic_segments(symbolic_root)
    neural_candidates = discover_neural_reliabilities(neural_root)

    missing_offsets = [x for x in symbolic_segments if x not in neural_candidates]
    if missing_offsets:
        raise FileNotFoundError(
            "The neural run has no reliability file for symbolic segment offsets: "
            + ", ".join(str(x) for x in missing_offsets)
        )

    print(f"[Info] symbolic root: {symbolic_root}")
    print(f"[Info] neural root:   {neural_root}")
    print(f"[Info] output:        {output_dir}")
    print(f"[Info] matched symbolic segments: {len(symbolic_segments)}")

    collected: dict[str, list[np.ndarray]] = {
        "ground_truth": [],
        "gt_grid_idx": [],
        "reliability_neural": [],
        "reliability_emission_oracle": [],
        "reliability_decoding_oracle": [],
        "trajectory_uniform": [],
        "trajectory_neural": [],
        "trajectory_emission_oracle": [],
        "trajectory_decoding_oracle": [],
    }
    segment_records: list[dict[str, Any]] = []
    segment_offsets: list[int] = []
    segment_lengths: list[int] = []
    grid_reference: Optional[np.ndarray] = None
    neighbor_reference: Optional[np.ndarray] = None

    expected_offset: Optional[int] = None

    for offset, symbolic_dir in symbolic_segments.items():
        print("\n" + "=" * 100)
        print(f"[Segment b{offset:06d}] {symbolic_dir}")
        print("=" * 100)

        result = process_segment(
            offset=offset,
            symbolic_dir=symbolic_dir,
            neural_candidates=neural_candidates[offset],
            symbolic_root=symbolic_root,
            num_sample=args.num_sample,
            device=device,
            temperature=args.temperature,
            margin_weight=args.margin_weight,
            concentration_weight=args.concentration_weight,
            search_steps=search_steps,
            max_passes_per_step=args.max_passes_per_step,
            decoding_oracle_init=args.decoding_oracle_init,
            Viterbi_Algorithm=Viterbi_Algorithm,
            soft_em_utils=soft_em_utils,
        )

        arrays = result["arrays"]
        meta = result["meta"]

        if expected_offset is not None and offset != expected_offset:
            raise RuntimeError(
                f"Segment gap or overlap: expected next block offset {expected_offset}, got {offset}."
            )
        expected_offset = offset + int(meta["num_blocks"])

        if grid_reference is None:
            grid_reference = result["grid"]
            neighbor_reference = result["neighbor"]
        else:
            if not np.array_equal(grid_reference, result["grid"]):
                raise ValueError(f"Grid differs at block offset {offset}.")
            if not np.array_equal(neighbor_reference, result["neighbor"]):
                raise ValueError(f"Neighbor matrix differs at block offset {offset}.")

        for key in collected:
            collected[key].append(arrays[key])

        segment_offsets.append(offset)
        segment_lengths.append(int(meta["t_total"]))
        segment_records.append(meta)

        seg_out = segments_out / f"b{offset:06d}"
        seg_out.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            np.save(seg_out / f"{name}.npy", array)
        save_json(meta, seg_out / "segment_manifest.json")

    assert grid_reference is not None and neighbor_reference is not None

    concat = {
        "ground_truth": np.concatenate(collected["ground_truth"], axis=0),
        "gt_grid_idx": np.concatenate(collected["gt_grid_idx"], axis=0),
        "reliability_neural": np.concatenate(collected["reliability_neural"], axis=1),
        "reliability_emission_oracle": np.concatenate(
            collected["reliability_emission_oracle"], axis=1
        ),
        "reliability_decoding_oracle": np.concatenate(
            collected["reliability_decoding_oracle"], axis=1
        ),
        "trajectory_uniform": np.concatenate(collected["trajectory_uniform"], axis=0),
        "trajectory_neural": np.concatenate(collected["trajectory_neural"], axis=0),
        "trajectory_emission_oracle": np.concatenate(
            collected["trajectory_emission_oracle"], axis=0
        ),
        "trajectory_decoding_oracle": np.concatenate(
            collected["trajectory_decoding_oracle"], axis=0
        ),
    }

    # Final exact shape checks. No truncation is allowed.
    t_total = concat["ground_truth"].shape[0]
    q = concat["reliability_neural"].shape[0]
    for key in (
        "reliability_neural",
        "reliability_emission_oracle",
        "reliability_decoding_oracle",
    ):
        if concat[key].shape != (q, t_total):
            raise RuntimeError(f"Final shape mismatch for {key}: {concat[key].shape}")
    for key in (
        "trajectory_uniform",
        "trajectory_neural",
        "trajectory_emission_oracle",
        "trajectory_decoding_oracle",
    ):
        if concat[key].shape != (t_total, 2):
            raise RuntimeError(f"Final shape mismatch for {key}: {concat[key].shape}")

    np.save(output_dir / "ground_truth_concat.npy", concat["ground_truth"])
    np.save(output_dir / "ground_truth_grid_idx_concat.npy", concat["gt_grid_idx"])
    np.save(output_dir / "grid.npy", grid_reference)
    np.save(output_dir / "neighbor_matrix.npy", neighbor_reference)
    np.save(output_dir / "segment_offsets.npy", np.asarray(segment_offsets, dtype=np.int64))
    np.save(output_dir / "segment_lengths.npy", np.asarray(segment_lengths, dtype=np.int64))

    np.save(output_dir / "reliability_neural_concat.npy", concat["reliability_neural"])
    np.save(
        output_dir / "reliability_emission_oracle_concat.npy",
        concat["reliability_emission_oracle"],
    )
    np.save(
        output_dir / "reliability_decoding_oracle_concat.npy",
        concat["reliability_decoding_oracle"],
    )

    np.save(output_dir / "trajectory_uniform_concat.npy", concat["trajectory_uniform"])
    np.save(output_dir / "trajectory_neural_concat.npy", concat["trajectory_neural"])
    np.save(
        output_dir / "trajectory_emission_oracle_concat.npy",
        concat["trajectory_emission_oracle"],
    )
    np.save(
        output_dir / "trajectory_decoding_oracle_concat.npy",
        concat["trajectory_decoding_oracle"],
    )

    manifest = {
        "symbolic_run_dir": str(symbolic_root),
        "neural_run_dir": str(neural_root),
        "output_dir": str(output_dir),
        "num_segments": len(segment_records),
        "segment_offsets": segment_offsets,
        "segment_lengths": segment_lengths,
        "num_sample": int(args.num_sample),
        "total_t": int(t_total),
        "q": int(q),
        "g": int(grid_reference.shape[0]),
        "oracle_parameters": {
            "temperature": float(args.temperature),
            "margin_weight": float(args.margin_weight),
            "concentration_weight": float(args.concentration_weight),
            "search_steps": search_steps,
            "max_passes_per_step": int(args.max_passes_per_step),
            "decoding_oracle_init": args.decoding_oracle_init,
            "emission_oracle_scope": "independent_block; AP z-score at each time",
            "decoding_oracle_scope": "independent_block; one AP bias vector per block",
            "decoding_objective": "block mean localization error",
            "decoding_protocol": "independent block-wise Viterbi",
        },
        "segments": segment_records,
        "saved_arrays": {name: list(array.shape) for name, array in concat.items()},
    }
    save_json(manifest, output_dir / "build_manifest.json")

    print("\n[Done] neural_boundary arrays were saved to:")
    print(output_dir)
    print(f"[Done] Q={q}, T_total={t_total}, segments={len(segment_records)}")
    print("Next run: python analysis/analyze_neural_boundary.py")


if __name__ == "__main__":
    main()
