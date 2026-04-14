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
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        self.load: Optional[LoadStage] = None
        self.repr: Optional[RepresentStage] = None
        self.proxy: Optional[ProxyStage] = None
        self.train: Optional[TrainStage] = None

        self.steps = 0

        # replay control
        self.replay_updates_per_step = int(config.get("NEURAL_REPLAY_UPDATES_PER_STEP", 2))
        self.replay_batch_size = int(config.get("NEURAL_REPLAY_BATCH_SIZE", 16))

        # publish control
        self.update_interval = int(config.get("NEURAL_UPDATE_INTERVAL", 10))
        self.min_updates_before_publish = int(config.get("NEURAL_MIN_UPDATES", 50))

        # simple gate to avoid collapsed model
        self.min_rel_std = float(config.get("NEURAL_MIN_REL_STD", 0.05))
        self.min_rel_gap = float(config.get("NEURAL_MIN_REL_GAP", 0.02))

        # debug
        self.save_debug = bool(config.get("SAVE_NEURAL_DEBUG", True))
        self.debug_dir = str(config.get("NEURAL_DEBUG_DIR", "output/neural_debug"))
        self.debug_save_interval = int(config.get("NEURAL_DEBUG_SAVE_INTERVAL", 1))

    def setup(self) -> None:
        self.load = LoadStage(self.config, self.device)
        self.repr = RepresentStage(self.config, self.device)
        self.proxy = ProxyStage(self.config, self.device)
        self.train = TrainStage(self.config, self.device)

        if self.save_debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    @property
    def model(self):
        if self.train is None:
            raise RuntimeError("Runtime not setup.")
        return self.train.model

    def run_step(self, neural_pkg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.load is None or self.repr is None or self.proxy is None or self.train is None:
            raise RuntimeError("Call setup() first.")

        # --------------------------------------------------
        # 1. append new sample into replay buffer
        # --------------------------------------------------
        self.load.append(neural_pkg)

        if not self.load.ready():
            return None

        # --------------------------------------------------
        # 2. train with replay (multiple updates)
        # --------------------------------------------------
        last_metrics = None

        for _ in range(self.replay_updates_per_step):
            batch_pkg = self.load.sample_batch(batch_size=self.replay_batch_size)
            if batch_pkg is None:
                continue

            aggregated_csi = batch_pkg["aggregated_csi"].to(self.device, non_blocking=True)
            emission = batch_pkg["emission_log_probs_qgt"].to(self.device, non_blocking=True)
            posterior = batch_pkg["posterior_gt"].to(self.device, non_blocking=True)

            pattern = self.repr.process(aggregated_csi)

            proxy_pkg = self.proxy.build(
                emission_log_probs_qgt=emission,
                posterior_gt=posterior,
            )

            last_metrics = self.train.step(
                pattern=pattern,
                target_score=proxy_pkg["expected_logprob"],
            )

        self.steps += 1

        # --------------------------------------------------
        # 3. debug saving
        # --------------------------------------------------
        if self.save_debug and (self.steps % self.debug_save_interval == 0):
            self._save_debug_tensors()

        # --------------------------------------------------
        # 4. publish decision (with simple gate)
        # --------------------------------------------------
        should_publish = (
            (self.steps >= self.min_updates_before_publish)
            and (self.steps % self.update_interval == 0)
        )

        if should_publish and last_metrics is not None:
            rel_std = last_metrics.get("pred_std", 0.0)
            rel_gap = last_metrics.get("top1_top2_gap", 0.0)

            # avoid collapsed reliability
            if rel_std > self.min_rel_std and rel_gap > self.min_rel_gap:
                return {
                    "type": "model_update",
                    "step": self.steps,
                    "state_dict": self.train.get_state_dict(),
                    "metrics": last_metrics,
                }

        return {
            "type": "training_step",
            "step": self.steps,
            "metrics": last_metrics,
        }

    def _save_debug_tensors(self) -> None:
        if self.train is None:
            return

        debug_tensors = self.train.get_debug_tensors()
        if not debug_tensors:
            return

        step_dir = os.path.join(self.debug_dir, f"step_{self.steps:06d}")
        os.makedirs(step_dir, exist_ok=True)

        for name, tensor in debug_tensors.items():
            if tensor is None:
                continue

            if isinstance(tensor, torch.Tensor):
                array = tensor.detach().cpu().numpy()
            else:
                array = np.asarray(tensor)

            np.save(os.path.join(step_dir, f"{name}.npy"), array)

    def save_checkpoint(self, filepath: str) -> None:
        if self.train is None:
            return

        checkpoint = {
            "state_dict": self.train.get_state_dict(),
            "optimizer_state": self.train.optimizer.state_dict(),
            "step": self.steps,
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
            self.steps = checkpoint.get("step", 0)

            print(f"[NeuralRuntime] Resumed from step {self.steps}")
            return True

        except Exception as e:
            print(f"[NeuralRuntime] Resume failed: {e}")
            return False