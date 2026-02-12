# engines/adapt_engine/stages/represent/stage.py

from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn.functional as F


class RepresentStage:
    """
    RepresentStage (Layer 0): Preprocessing for Shared Encoder.
    
    Transforms (B, Q, T, N, M) -> (B*Q*T, N*2, M)
    This allows the ResNet to learn a unified topology space for any AP.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.eps = 1e-6

    def process(self, raw_csi: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, Q, T, N, M)
        Output: (Total_Samples, Channels, Length) = (B*Q*T, N*2, M)
        """
        # Ensure data is on GPU
        if raw_csi.device != self.device:
            raw_csi = raw_csi.to(self.device, non_blocking=True)

        # 1. Log-Magnitude Feature (Signal Strength)
        # Shape: (B, Q, T, N, M)
        mag = raw_csi.abs()
        log_mag = torch.log10(mag + self.eps)

        # 2. Diff-Phase Feature (SFO Removal)
        # Shape: (B, Q, T, N, M)
        phase = raw_csi.angle()
        diff_phase = torch.diff(phase, dim=-1, prepend=phase[..., 0:1])

        # 3. Stack Features
        # Shape: (B, Q, T, N, M, 2)
        x = torch.stack([log_mag, diff_phase], dim=-1)

        # 4. Reshape for 1D-CNN Shared Encoder
        B, Q, T, N, M, _ = x.shape

        # Permute: Move M to end (Length), N & 2 to middle
        # (B, Q, T, N, M, 2) -> (B, Q, T, N, 2, M)
        x = x.permute(0, 1, 2, 3, 5, 4)

        # Flatten B, Q, T into Batch dim (Independent Samples)
        # Flatten N, 2 into Channel dim (Spatial/Feature info)
        # Final Shape: (B*Q*T, N*2, M)
        x_flat = x.reshape(B * Q * T, N * 2, M)

        # 5. Instance Normalization
        # Normalizes each AP/Time instance independently
        return F.instance_norm(x_flat)