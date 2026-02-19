# engines/adapt_engine/runtime.py

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import os

from engines.adapt_engine.stages.load.stage import LoadStage
from engines.adapt_engine.stages.represent.stage import RepresentStage
from engines.adapt_engine.stages.train.stage import TrainStage


class AdaptRuntime:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        # Stages
        self.load: Optional[LoadStage] = None
        self.repr: Optional[RepresentStage] = None
        self.train: Optional[TrainStage] = None

        # Scheduling
        self.update_interval = int(config.get("ADAPT_UPDATE_INTERVAL", 10))
        self.min_updates_before_publish = int(config.get("ADAPT_MIN_UPDATES", 50))
        self.steps = 0

    def setup(self) -> None:
        self.load = LoadStage(self.config, self.device)
        self.repr = RepresentStage(self.config, self.device)
        self.train = TrainStage(self.config, self.device)
    
    @property
    def model(self):
        if self.train is None:
            raise RuntimeError("Runtime not setup. Call setup() first.")
        return self.train.model

    def run_step(self, raw_csi_cpu: torch.Tensor) -> Optional[Dict[str, Any]]:
        if self.load is None:
            raise RuntimeError("AdaptRuntime.setup() must be called before run_step()")

        # Phase 1: Load (Data Accumulation)
        batch_tensor = self.load.accumulate(raw_csi_cpu)

        if batch_tensor is None:
            return None

        # Phase 2: Represent (Physics Transformation)
        batch_gpu = batch_tensor.to(self.device, non_blocking=True)
        features = self.repr.process(batch_gpu)

        # Phase 3: Train (Model Learning)
        metrics = self.train.step(features)
        
        self.steps += 1

        # Phase 4: Publish Decision
        should_publish = (
            (self.steps > self.min_updates_before_publish) and
            (self.steps % self.update_interval == 0)
        )

        if should_publish:
            state_dict = self.train.get_state_dict()
            
            return {
                "type": "model_update",
                "step": self.steps,
                "model_state": state_dict,
                "metrics": metrics
            }
        
        return None
    
    def save_checkpoint(self, filepath: str) -> None:
        """Saves model, optimizer, steps, and config."""
        if self.train is None: 
            return
        else:
            state_dict = self.train.get_state_dict()

        checkpoint = {
            "model_state": state_dict,
            "optimizer_state": self.train.optimizer.state_dict(),
            "step": self.steps,
            "config": self.config
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> bool:
        """Resumes training from a file."""
        if self.train is None: return False
        
        if not os.path.exists(filepath):
            print(f"[AdaptRuntime] No checkpoint at {filepath}, starting fresh.")
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            
            # Restore state
            self.train.model.load_state_dict(checkpoint["model_state"])
            self.train.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.steps = checkpoint.get("step", 0)

            print(f"[AdaptRuntime] Resumed from step {self.steps}.")
            return True
        except Exception as e:
            print(f"[AdaptRuntime] Resume failed: {e}")
            return False