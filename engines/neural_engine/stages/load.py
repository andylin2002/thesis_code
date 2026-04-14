# engines/neural_engine/stages/load.py

import random
from typing import Dict, List, Optional

import torch


class LoadStage:
    """Store teacher-labeled samples and serve replay batches."""

    def __init__(self, config: Dict, device: torch.device):
        self.device = device

        self.batch_size = int(config.get("NEURAL_BATCH_SIZE", 16))
        self.buffer_capacity = int(config.get("NEURAL_REPLAY_BUFFER_SIZE", 512))
        self.min_buffer_size = int(
            config.get("NEURAL_MIN_BUFFER_SIZE", self.batch_size)
        )
        self.include_latest = bool(config.get("NEURAL_INCLUDE_LATEST", True))

        self.buffer: List[Dict[str, torch.Tensor]] = []

    def _validate_pkg(self, neural_pkg: Dict[str, torch.Tensor]) -> None:
        if not isinstance(neural_pkg, dict):
            raise ValueError("neural_pkg must be a dict")

        required_keys = ["aggregated_csi", "emission_log_probs_qgt", "posterior_gt"]
        for key in required_keys:
            if key not in neural_pkg:
                raise KeyError(f"Missing key in neural_pkg: {key}")

        if not isinstance(neural_pkg["aggregated_csi"], torch.Tensor):
            raise TypeError("aggregated_csi must be a torch.Tensor")
        if not isinstance(neural_pkg["emission_log_probs_qgt"], torch.Tensor):
            raise TypeError("emission_log_probs_qgt must be a torch.Tensor")
        if not isinstance(neural_pkg["posterior_gt"], torch.Tensor):
            raise TypeError("posterior_gt must be a torch.Tensor")

    def _clone_to_cpu(self, neural_pkg: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "aggregated_csi": neural_pkg["aggregated_csi"].detach().cpu().clone(),
            "emission_log_probs_qgt": neural_pkg["emission_log_probs_qgt"].detach().cpu().clone(),
            "posterior_gt": neural_pkg["posterior_gt"].detach().cpu().clone(),
        }

    def _stack_indices(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        return {
            "aggregated_csi": torch.stack(
                [self.buffer[i]["aggregated_csi"] for i in indices], dim=0
            ),
            "emission_log_probs_qgt": torch.stack(
                [self.buffer[i]["emission_log_probs_qgt"] for i in indices], dim=0
            ),
            "posterior_gt": torch.stack(
                [self.buffer[i]["posterior_gt"] for i in indices], dim=0
            ),
        }

    def append(self, neural_pkg: Dict[str, torch.Tensor]) -> int:
        self._validate_pkg(neural_pkg)
        item = self._clone_to_cpu(neural_pkg)

        self.buffer.append(item)

        if self.buffer_capacity > 0 and len(self.buffer) > self.buffer_capacity:
            overflow = len(self.buffer) - self.buffer_capacity
            self.buffer = self.buffer[overflow:]

        return len(self.buffer)

    def ready(self, min_size: Optional[int] = None) -> bool:
        required = self.min_buffer_size if min_size is None else int(min_size)
        return len(self.buffer) >= required

    def size(self) -> int:
        return len(self.buffer)

    def sample_batch(
        self,
        batch_size: Optional[int] = None,
        include_latest: Optional[bool] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.buffer:
            return None

        batch_n = self.batch_size if batch_size is None else int(batch_size)
        use_latest = self.include_latest if include_latest is None else bool(include_latest)

        if len(self.buffer) < batch_n:
            return None

        latest_idx = len(self.buffer) - 1
        all_indices = list(range(len(self.buffer)))

        if use_latest:
            candidate_indices = all_indices[:-1]
            sample_n = batch_n - 1

            if sample_n < 0:
                raise ValueError("batch_size must be >= 1")

            if sample_n == 0:
                chosen = [latest_idx]
            else:
                chosen = random.sample(candidate_indices, k=sample_n)
                chosen.append(latest_idx)
        else:
            chosen = random.sample(all_indices, k=batch_n)

        return self._stack_indices(chosen)

    def sample_recent_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        batch_n = self.batch_size if batch_size is None else int(batch_size)
        if len(self.buffer) < batch_n:
            return None

        start = len(self.buffer) - batch_n
        indices = list(range(start, len(self.buffer)))
        return self._stack_indices(indices)

    def accumulate(self, neural_pkg: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Backward-compatible entry point.
        Append new sample, then return one replay batch if buffer is ready.
        """
        self.append(neural_pkg)

        if not self.ready():
            return None

        return self.sample_batch()

    def state_dict(self) -> Dict[str, int]:
        return {
            "buffer_size": len(self.buffer),
            "batch_size": self.batch_size,
            "buffer_capacity": self.buffer_capacity,
            "min_buffer_size": self.min_buffer_size,
        }