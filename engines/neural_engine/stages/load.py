# engines/neural_engine/stages/load.py

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import torch


class LoadStage:
    """Collect new blocks and build sliding window batches for neural training."""

    def __init__(self, config: Dict, device: torch.device):
        self.device = device

        self.B = int(config.get("NEURAL_BATCH_SIZE", 32))
        self.W = int(config.get("NEURAL_WINDOW_SIZE", 16))
        self.window_stride = int(config.get("NEURAL_WINDOW_STRIDE", 1))
        self.buffer_capacity = int(
            config.get("NEURAL_SEQUENCE_BUFFER_CAPACITY", max(self.B * 4, self.W * 4))
        )

        if self.B <= 0:
            raise ValueError("NEURAL_BATCH_SIZE must be > 0")
        if self.W <= 0:
            raise ValueError("NEURAL_WINDOW_SIZE must be > 0")
        if self.W > self.B:
            raise ValueError("NEURAL_WINDOW_SIZE must be <= NEURAL_BATCH_SIZE")
        if self.window_stride <= 0:
            raise ValueError("NEURAL_WINDOW_STRIDE must be > 0")
        if self.window_stride > self.W:
            raise ValueError("NEURAL_WINDOW_STRIDE must be <= NEURAL_WINDOW_SIZE")
        if self.buffer_capacity < self.B:
            raise ValueError("NEURAL_SEQUENCE_BUFFER_CAPACITY must be >= NEURAL_BATCH_SIZE")

        self.pending_new_blocks: Deque[Dict[str, Optional[torch.Tensor]]] = deque(maxlen=self.B)
        self.history_tail_blocks: Deque[Dict[str, Optional[torch.Tensor]]] = deque(maxlen=max(self.W - 1, 0))

        self.num_total_blocks = 0
        self.num_emitted_batches = 0

    def append(self, neural_pkg: Dict[str, torch.Tensor]) -> int:
        self._validate_pkg(neural_pkg)
        self.pending_new_blocks.append(self._clone_to_cpu(neural_pkg))
        self.num_total_blocks += 1
        return len(self.pending_new_blocks)

    def ready(self) -> bool:
        return len(self.pending_new_blocks) == self.B

    def size(self) -> int:
        return len(self.pending_new_blocks)

    def get_window(self) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        if not self.ready():
            return None

        new_blocks = list(self.pending_new_blocks)
        tail_blocks = list(self.history_tail_blocks)

        sequence_blocks = tail_blocks + new_blocks
        num_sequence_blocks = len(sequence_blocks)

        if num_sequence_blocks < self.W:
            raise RuntimeError(
                f"Not enough blocks to build one window: sequence={num_sequence_blocks}, W={self.W}"
            )

        start_indices = list(range(0, num_sequence_blocks - self.W + 1, self.window_stride))

        windows_aggregated_csi: List[torch.Tensor] = []
        windows_emission_log_probs_qgt: List[torch.Tensor] = []
        windows_posterior_gt: List[torch.Tensor] = []

        has_full_posterior = all(block["posterior_gt"] is not None for block in sequence_blocks)

        for start in start_indices:
            window_blocks = sequence_blocks[start:start + self.W]

            aggregated_csi = torch.stack(
                [block["aggregated_csi"] for block in window_blocks],
                dim=0,
            )  # [W,Q,T,N,M]

            emission_log_probs_qgt = torch.stack(
                [block["emission_log_probs_qgt"] for block in window_blocks],
                dim=0,
            )  # [W,Q,G,T]

            windows_aggregated_csi.append(aggregated_csi)
            windows_emission_log_probs_qgt.append(emission_log_probs_qgt)

            if has_full_posterior:
                posterior_gt = torch.stack(
                    [block["posterior_gt"] for block in window_blocks],
                    dim=0,
                )  # [W,G,T]
                windows_posterior_gt.append(posterior_gt)

        batch_aggregated_csi = torch.stack(
            windows_aggregated_csi,
            dim=0,
        )  # [S,W,Q,T,N,M]

        batch_emission_log_probs_qgt = torch.stack(
            windows_emission_log_probs_qgt,
            dim=0,
        )  # [S,W,Q,G,T]

        if has_full_posterior:
            batch_posterior_gt = torch.stack(
                windows_posterior_gt,
                dim=0,
            )  # [S,W,G,T]
        else:
            batch_posterior_gt = None

        self._update_history_tail(new_blocks)
        self.pending_new_blocks.clear()
        self.num_emitted_batches += 1

        return {
            "aggregated_csi": batch_aggregated_csi,
            "emission_log_probs_qgt": batch_emission_log_probs_qgt,
            "posterior_gt": batch_posterior_gt,
            "num_windows": batch_aggregated_csi.shape[0],
            "window_size": self.W,
            "batch_size": self.B,
        }

    def state_dict(self) -> Dict[str, int]:
        return {
            "pending_new_blocks": len(self.pending_new_blocks),
            "history_tail_blocks": len(self.history_tail_blocks),
            "num_total_blocks": self.num_total_blocks,
            "num_emitted_batches": self.num_emitted_batches,
            "batch_size_B": self.B,
            "window_size_W": self.W,
            "window_stride": self.window_stride,
            "buffer_capacity": self.buffer_capacity,
        }

    def _update_history_tail(
        self,
        new_blocks: List[Dict[str, Optional[torch.Tensor]]],
    ) -> None:
        self.history_tail_blocks.clear()

        if self.W <= 1:
            return

        tail_length = self.W - 1
        tail_blocks = new_blocks[-tail_length:]

        for block in tail_blocks:
            self.history_tail_blocks.append(block)

    def _validate_pkg(self, neural_pkg: Dict[str, torch.Tensor]) -> None:
        if not isinstance(neural_pkg, dict):
            raise TypeError("neural_pkg must be a dict")

        required_keys = ["aggregated_csi", "emission_log_probs_qgt"]
        for key in required_keys:
            if key not in neural_pkg:
                raise KeyError(f"Missing key in neural_pkg: {key}")

        aggregated_csi = neural_pkg["aggregated_csi"]
        emission_log_probs_qgt = neural_pkg["emission_log_probs_qgt"]
        posterior_gt = neural_pkg.get("posterior_gt", None)

        if not isinstance(aggregated_csi, torch.Tensor):
            raise TypeError("aggregated_csi must be a torch.Tensor")
        if not isinstance(emission_log_probs_qgt, torch.Tensor):
            raise TypeError("emission_log_probs_qgt must be a torch.Tensor")
        if posterior_gt is not None and not isinstance(posterior_gt, torch.Tensor):
            raise TypeError("posterior_gt must be a torch.Tensor or None")

        if aggregated_csi.ndim != 4:
            raise ValueError(
                f"Expected aggregated_csi shape [Q,T,N,M], got {tuple(aggregated_csi.shape)}"
            )

        if emission_log_probs_qgt.ndim != 3:
            raise ValueError(
                f"Expected emission_log_probs_qgt shape [Q,G,T], got {tuple(emission_log_probs_qgt.shape)}"
            )

        if posterior_gt is not None and posterior_gt.ndim != 2:
            raise ValueError(
                f"Expected posterior_gt shape [G,T], got {tuple(posterior_gt.shape)}"
            )

        q_csi, t_csi, _, _ = aggregated_csi.shape
        q_epd, g_epd, t_epd = emission_log_probs_qgt.shape

        if q_csi != q_epd:
            raise ValueError(
                f"AP dimension mismatch: aggregated_csi Q={q_csi}, emission Q={q_epd}"
            )

        if t_csi != t_epd:
            raise ValueError(
                f"Time dimension mismatch: aggregated_csi T={t_csi}, emission T={t_epd}"
            )

        if posterior_gt is not None:
            g_post, t_post = posterior_gt.shape

            if g_post != g_epd:
                raise ValueError(
                    f"Grid dimension mismatch: emission G={g_epd}, posterior G={g_post}"
                )

            if t_post != t_epd:
                raise ValueError(
                    f"Time dimension mismatch: emission T={t_epd}, posterior T={t_post}"
                )

    def _clone_to_cpu(
        self,
        neural_pkg: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        aggregated_csi = neural_pkg["aggregated_csi"].detach().cpu().clone()
        emission_log_probs_qgt = neural_pkg["emission_log_probs_qgt"].detach().cpu().clone()

        posterior_gt = neural_pkg.get("posterior_gt", None)
        if posterior_gt is not None:
            posterior_gt = posterior_gt.detach().cpu().clone()

        return {
            "aggregated_csi": aggregated_csi,
            "emission_log_probs_qgt": emission_log_probs_qgt,
            "posterior_gt": posterior_gt,
        }