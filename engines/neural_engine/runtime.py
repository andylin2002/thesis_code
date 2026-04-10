# engines/neural_engine/runtime.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from engines.neural_engine.stages.load import LoadStage
from engines.neural_engine.stages.represent import RepresentStage
from engines.neural_engine.stages.train import TrainStage


class NeuralRuntime:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        self.load: Optional[LoadStage] = None
        self.repr: Optional[RepresentStage] = None
        self.train: Optional[TrainStage] = None

        self.update_interval = int(config.get("NEURAL_UPDATE_INTERVAL", 10))
        self.min_updates_before_publish = int(config.get("NEURAL_MIN_UPDATES", 50))
        self.steps = 0

        self.save_debug = bool(config.get("SAVE_NEURAL_DEBUG", True))
        self.debug_dir = str(config.get("NEURAL_DEBUG_DIR", "output/neural_debug"))
        self.debug_save_interval = int(config.get("NEURAL_DEBUG_SAVE_INTERVAL", 1))

    def setup(self) -> None:
        self.load = LoadStage(self.config, self.device)
        self.repr = RepresentStage(self.config, self.device)
        self.train = TrainStage(
            self.config,
            self.device,
            self.repr.encoder,
        )

        if self.save_debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    @property
    def model(self):
        if self.train is None:
            raise RuntimeError("Runtime not setup. Call setup() first.")
        return self.train.encoder

    def run_step(self, raw_csi_cpu: torch.Tensor) -> Optional[Dict[str, Any]]:
        if self.load is None or self.repr is None or self.train is None:
            raise RuntimeError("NeuralRuntime.setup() must be called before run_step()")

        batch_tensor = self.load.accumulate(raw_csi_cpu)
        if batch_tensor is None:
            return None

        batch_gpu = batch_tensor.to(self.device, non_blocking=True)

        input_features = self.repr._build_input_features(batch_gpu)   # [B, Q, T, C, M]
        metrics = self.train.step(input_features)
        self.steps += 1

        if self.save_debug and (self.steps % self.debug_save_interval == 0):
            self._save_debug_tensors()

        should_publish = (
            (self.steps > self.min_updates_before_publish)
            and (self.steps % self.update_interval == 0)
        )

        if should_publish:
            state_dict = self.train.get_state_dict()
            return {
                "type": "model_update",
                "step": self.steps,
                "state_dict": state_dict,
                "metrics": metrics,
            }

        return {
            "type": "training_step",
            "step": self.steps,
            "metrics": metrics,
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
            print(f"[NeuralRuntime] No checkpoint at {filepath}, starting fresh.")
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

            state_dict = checkpoint["state_dict"]
            self.train.encoder.load_state_dict(state_dict["encoder"])
            self.train.reliability_head.load_state_dict(state_dict["reliability_head"])

            self.train.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.steps = checkpoint.get("step", 0)

            print(f"[NeuralRuntime] Resumed from step {self.steps}.")
            return True

        except Exception as e:
            print(f"[NeuralRuntime] Resume failed: {e}")
            return False