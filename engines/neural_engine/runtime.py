# engines/neural_engine/runtime.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from engines.neural_engine.stages.load import LoadStage
from engines.neural_engine.stages.represent import RepresentStage
from engines.neural_engine.stages.proxy import ProxyStage
from engines.neural_engine.stages.train import TrainStage


class NeuralRuntime:
    """online long-horizon proxy."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        self.load: Optional[LoadStage] = None
        self.repr: Optional[RepresentStage] = None
        self.proxy: Optional[ProxyStage] = None
        self.train: Optional[TrainStage] = None

        self.steps = 0
        self.num_updates = 0

        self.update_interval = int(config.get("NEURAL_UPDATE_INTERVAL", 10))
        self.min_updates_before_publish = int(config.get("NEURAL_MIN_UPDATES", 50))

        self.save_debug = bool(config.get("SAVE_NEURAL_DEBUG", False))
        self.debug_dir = str(config.get("NEURAL_DEBUG_DIR", "output/neural_debug"))
        self.debug_save_interval = int(config.get("NEURAL_DEBUG_SAVE_INTERVAL", 50))
        self.updates_per_batch = int(config.get("NEURAL_UPDATES_PER_BATCH", 1))

        self.neighbor_index_matrix: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.load = LoadStage(self.config, self.device)
        self.repr = RepresentStage(self.config, self.device)
        self.proxy = ProxyStage(self.config, self.device)
        self.train = TrainStage(self.config, self.device)

        self.neighbor_index_matrix = self._load_neighbor_index_matrix()

        if self.save_debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    @property
    def model(self):
        if self.train is None:
            raise RuntimeError("Runtime not setup.")
        return self.train.model

    def run_step(self, neural_pkg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if (
            self.load is None
            or self.repr is None
            or self.proxy is None
            or self.train is None
        ):
            raise RuntimeError("Call setup() first.")

        if self.neighbor_index_matrix is None:
            raise RuntimeError("neighbor_index_matrix is not initialized")

        self.steps += 1
        self.load.append(neural_pkg)

        if not self.load.ready():
            return None

        batch_pkg = self.load.get_window()
        if batch_pkg is None:
            return None

        aggregated_csi_swqtnm = batch_pkg["aggregated_csi"]          # [S,W,Q,T,N,M]
        emission_swqgt = batch_pkg["emission_log_probs_qgt"]         # [S,W,Q,G,T]

        aggregated_csi_swqtnm = aggregated_csi_swqtnm.to(self.device, non_blocking=True)
        emission_swqgt = emission_swqgt.to(self.device, non_blocking=True)

        s, w, q, t, n, m = aggregated_csi_swqtnm.shape
        s2, w2, q2, g, t2 = emission_swqgt.shape

        if s != s2 or w != w2 or q != q2 or t != t2:
            raise ValueError(
                "Shape mismatch between aggregated_csi and emission_log_probs_qgt: "
                f"aggregated_csi={tuple(aggregated_csi_swqtnm.shape)}, "
                f"emission={tuple(emission_swqgt.shape)}"
            )

        aggregated_csi_bqtnm = aggregated_csi_swqtnm.reshape(s * w, q, t, n, m)
        pattern_bqtcm = self.repr.process(aggregated_csi_bqtnm)      # [S*W,Q,T,C,M]

        proxy_pkg = self.proxy.build(
            emission_log_probs_qgt=emission_swqgt,
            neighbor_index_matrix=self.neighbor_index_matrix,
        )

        target_score_swqt = proxy_pkg["target_score"]                # [S,W,Q,T]
        target_score_bqt = target_score_swqt.reshape(s * w, q, t)   # [S*W,Q,T]

        metrics = None
        for _ in range(self.updates_per_batch):
            metrics = self.train.step(
                pattern=pattern_bqtcm,
                target_score=target_score_bqt
            )

        self.num_updates += 1

        if self.save_debug and (self.num_updates % self.debug_save_interval == 0):
            debug_tensors = self.train.get_debug_tensors()
            self._save_debug_tensors(
                pattern=pattern_bqtcm,
                target_score=target_score_bqt,
                pred_logits=debug_tensors.get("pred_logits"),
                pred_prob=debug_tensors.get("pred_prob"),
                teacher_prob=debug_tensors.get("teacher_prob"),
                embedding=debug_tensors.get("embedding"),
                num_windows=s,
                window_size=w,
            )

        should_publish = (
            self.num_updates >= self.min_updates_before_publish
            and self.num_updates % self.update_interval == 0
        )

        if should_publish:
            return {
                "type": "model_update",
                "step": self.steps,
                "num_updates": self.num_updates,
                "state_dict": self.train.get_state_dict(),
                "metrics": metrics,
            }

        return {
            "type": "training_step",
            "step": self.steps,
            "num_updates": self.num_updates,
            "metrics": metrics,
        }

    def _load_neighbor_index_matrix(self) -> torch.Tensor:
        output_dir = str(self.config.get("OUTPUT_DIR", "output"))
        neighbor_path = os.path.join(output_dir, "neighbor_matrix.npy")

        if not os.path.exists(neighbor_path):
            raise FileNotFoundError(f"neighbor_matrix.npy not found: {neighbor_path}")

        neighbor = np.load(neighbor_path)
        if neighbor.ndim != 2:
            raise ValueError(
                f"Expected neighbor_matrix with shape [G,K], got {tuple(neighbor.shape)}"
            )

        return torch.from_numpy(neighbor.astype(np.int64)).to(self.device)

    def _save_debug_tensors(
        self,
        pattern: torch.Tensor,
        target_score: torch.Tensor,
        pred_logits: Optional[torch.Tensor],
        pred_prob: torch.Tensor,
        teacher_prob: torch.Tensor,
        embedding: Optional[torch.Tensor],
        num_windows: int,
        window_size: int,
    ) -> None:
        step_dir = os.path.join(self.debug_dir, f"step_{self.num_updates:06d}")
        os.makedirs(step_dir, exist_ok=True)

        np.save(
            os.path.join(step_dir, "target_score_bqt.npy"),
            target_score.detach().cpu().numpy(),
        )

        if pred_logits is not None:
            np.save(
                os.path.join(step_dir, "pred_logits_btq.npy"),
                pred_logits.detach().cpu().numpy(),
            )

        if pred_prob is not None:
            np.save(
                os.path.join(step_dir, "pred_prob_btq.npy"),
                pred_prob.detach().cpu().numpy(),
            )

        if teacher_prob is not None:
            np.save(
                os.path.join(step_dir, "teacher_prob_btq.npy"),
                teacher_prob.detach().cpu().numpy(),
            )

        if embedding is not None:
            np.save(
                os.path.join(step_dir, "embedding_be.npy"),
                embedding.detach().cpu().numpy(),
            )

        pattern_mean = pattern.detach().mean(dim=(-1, -2)).cpu().numpy()
        np.save(
            os.path.join(step_dir, "pattern_mean_bqt.npy"),
            pattern_mean,
        )

        meta = {
            "num_windows": num_windows,
            "window_size": window_size,
            "num_block_samples": int(target_score.shape[0]),
        }
        np.save(
            os.path.join(step_dir, "meta.npy"),
            meta,
            allow_pickle=True,
        )

    def save_checkpoint(self, filepath: str) -> None:
        if self.train is None:
            return

        checkpoint = {
            "state_dict": self.train.get_state_dict(),
            "optimizer_state": self.train.optimizer.state_dict(),
            "step": self.steps,
            "num_updates": self.num_updates,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> bool:
        if self.train is None:
            return False

        if not os.path.exists(filepath):
            print(f"[NeuralRuntime] No checkpoint at {filepath}")
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

            state_dict = checkpoint["state_dict"]
            self.train.model.load_state_dict(state_dict["model"])
            self.train.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.steps = int(checkpoint.get("step", 0))
            self.num_updates = int(checkpoint.get("num_updates", 0))

            print(f"[NeuralRuntime] Resumed from step {self.steps}, updates {self.num_updates}")
            return True

        except Exception as e:
            print(f"[NeuralRuntime] Resume failed: {e}")
            return False