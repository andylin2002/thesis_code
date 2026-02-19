# engines/adapt_engine/stages/represent/stage.py

from __future__ import annotations
from typing import Any, Dict
import torch

class RepresentStage:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.c_max = int(config.get("C_MAX_TAPS", 16))
        self.data_aug = config.get("DATA_AUG", False)
        self.q = float(config.get("AUG_SUB_RATIO", 0.2))

    def process(self, raw_csi: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, Q, T, N, M)
        Output: (B*Q*T, D)
        """
        if raw_csi.device != self.device:
            raw_csi = raw_csi.to(self.device, non_blocking=True)

        # Apply Data Augmentation if enabled
        if self.data_aug:
            raw_csi = self._apply_aug(raw_csi)

        # 1. Inverse DFT to Delay Domain
        h_delay = torch.fft.ifft(raw_csi, dim=-1)

        # 2. Truncate taps to retain large-scale fading
        h_trunc = h_delay[..., :self.c_max].abs()

        # 3. Reshape and L2 Normalization
        B, Q, T, N, C = h_trunc.shape
        f_flat = h_trunc.reshape(B * Q * T, -1)
        norm = torch.norm(f_flat, p=2, dim=-1, keepdim=True) + 1e-8
        return f_flat / norm

    def _apply_aug(self, x: torch.Tensor) -> torch.Tensor:
        """ Implement random hardware abstraction and noise injection """
        B, Q, T, N, M = x.shape
        
        # 1. Randomly deactivate antennas
        for b in range(B):
            for q_idx in range(Q):
                num_keep = torch.randint(1, N + 1, (1,)).item()
                idx = torch.randperm(N)[:num_keep]
                mask = torch.zeros(N, device=self.device)
                mask[idx] = 1.0
                x[b, q_idx] *= mask.view(1, N, 1)

        # 2. Randomly remove subcarrier bands
        num_remove = int(M * self.q)
        if num_remove > 0:
            start = torch.randint(0, M - num_remove + 1, (1,)).item()
            x[..., start : start + num_remove] = 0.0

        # 3. Add Gaussian noise
        return x + torch.randn_like(x) * 0.01