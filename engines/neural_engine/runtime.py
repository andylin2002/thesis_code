# engines/neural_engine/runtime.py

from __future__ import annotations
from typing import Any, Dict, Optional

import os
import numpy as np
import torch

from engines.neural_engine.stages.load import LoadStage
from engines.neural_engine.stages.represent import RepresentStage
from engines.neural_engine.stages.train import TrainStage


class NeuralRuntime:
    """
    Neural runtime for online / asynchronous self-supervised neural training.

    Pipeline:
        1. LoadStage      : accumulate raw CSI blocks into a training batch
        2. RepresentStage : preprocess CSI, encode spatial-temporal latent features,
                            and compute AP-wise relative probabilities
        3. TrainStage     : perform one self-supervised training step
        4. Publish        : periodically export updated model state_dict to symbolic engine

    Debug functionality:
        Save intermediate neural tensors to disk for diagnosis, including:
            - pred_error
            - target_probs
            - ap_probs
            - scores
            - score_std
            - prob_gap
            - top1_ap
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        # Stages
        self.load: Optional[LoadStage] = None
        self.repr: Optional[RepresentStage] = None
        self.train: Optional[TrainStage] = None

        # Scheduling
        self.update_interval = int(config.get("NEURAL_UPDATE_INTERVAL", 10))
        self.min_updates_before_publish = int(config.get("NEURAL_MIN_UPDATES", 50))
        self.steps = 0

        # Debug saving
        self.save_debug = bool(config.get("SAVE_NEURAL_DEBUG", True))
        self.debug_dir = str(config.get("NEURAL_DEBUG_DIR", "output/neural_debug"))
        self.debug_save_interval = int(config.get("NEURAL_DEBUG_SAVE_INTERVAL", 1))

    def setup(self) -> None:
        """
        Build all stages and prepare debug directory if enabled.
        """
        self.load = LoadStage(self.config, self.device)
        self.repr = RepresentStage(self.config, self.device)
        self.train = TrainStage(
            self.config,
            self.device,
            self.repr.encoder,
            self.repr.attention,
        )

        if self.save_debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    @property
    def model(self):
        """
        Keep compatibility with any existing code that expects runtime.model.
        """
        if self.train is None:
            raise RuntimeError("Runtime not setup. Call setup() first.")
        return self.train.encoder

    def run_step(self, raw_csi_cpu: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Run one runtime step.

        Returns:
            None
                if LoadStage has not yet accumulated enough CSI blocks into a batch

            {
                "type": "training_step",
                "step": int,
                "metrics": {...}
            }

            or

            {
                "type": "model_update",
                "step": int,
                "state_dict": {...},
                "metrics": {...}
            }
        """
        if self.load is None or self.repr is None or self.train is None:
            raise RuntimeError("NeuralRuntime.setup() must be called before run_step()")

        # ------------------------------------------------------------
        # Phase 1: Load (batch accumulation)
        # ------------------------------------------------------------
        batch_tensor = self.load.accumulate(raw_csi_cpu)
        if batch_tensor is None:
            return None

        # ------------------------------------------------------------
        # Phase 2: Represent
        # ------------------------------------------------------------
        batch_gpu = batch_tensor.to(self.device, non_blocking=True)

        # features: [B, T, Q, D]
        # ap_probs: [B, T, Q]
        #
        # We keep ap_probs from RepresentStage for interface consistency / possible
        # future diagnostics, although TrainStage will recompute probs_t on z_t to
        # access raw scores as well.
        features, ap_probs = self.repr.process(batch_gpu, return_scores=False)

        # ------------------------------------------------------------
        # Phase 3: Train
        # ------------------------------------------------------------
        metrics = self.train.step(features, ap_probs)
        self.steps += 1

        # ------------------------------------------------------------
        # Phase 3.5: Save debug tensors (optional)
        # ------------------------------------------------------------
        if self.save_debug and (self.steps % self.debug_save_interval == 0):
            self._save_debug_tensors()

        # ------------------------------------------------------------
        # Phase 4: Publish model update if scheduled
        # ------------------------------------------------------------
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
        """
        Save the latest debug tensors from TrainStage into:
            <debug_dir>/step_xxxxxx/*.npy

        Expected tensors from train.get_debug_tensors():
            pred_error   : [B, T-1, Q]
            target_probs : [B, T-1, Q]
            ap_probs     : [B, T-1, Q]
            scores       : [B, T-1, Q]
            score_std    : [B, T-1]
            prob_gap     : [B, T-1]
            top1_ap      : [B, T-1]
        """
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
        """
        Save:
            - encoder / attention / predictor state dicts
            - optimizer state
            - current training step
            - config
        """
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
        """
        Resume training from a checkpoint file.

        Returns:
            True  if successfully resumed
            False otherwise

        NOTE:
            Since you are currently changing the neural architecture, old checkpoints
            may fail to load or may bias the model toward an old uniform solution.
            For architecture validation, starting fresh is strongly recommended.
        """
        if self.train is None:
            return False

        if not os.path.exists(filepath):
            print(f"[NeuralRuntime] No checkpoint at {filepath}, starting fresh.")
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

            state_dict = checkpoint["state_dict"]
            self.train.encoder.load_state_dict(state_dict["encoder"])
            self.train.attention.load_state_dict(state_dict["attention"])
            self.train.predictor.load_state_dict(state_dict["predictor"])

            self.train.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.steps = checkpoint.get("step", 0)

            print(f"[NeuralRuntime] Resumed from step {self.steps}.")
            return True

        except Exception as e:
            print(f"[NeuralRuntime] Resume failed: {e}")
            return False