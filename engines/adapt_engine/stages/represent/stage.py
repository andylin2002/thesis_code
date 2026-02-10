# engines/adapt_engine/stages/represent/stage.py

from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn.functional as F


class RepresentStage:
    """
    RepresentStage (Layer 0): Geometry-Preserving CSI Representation.

    Input Shape:  (B, Q, T, N, M)  <- N=Antennas, M=Subcarriers
    Output Shape: (B, Channels, M) <- Preserves Subcarrier axis for 1D-CNN
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.eps = 1e-6

    def process(self, raw_csi: torch.Tensor) -> torch.Tensor:
        """
        Transforms raw complex CSI into hardware-invariant features.
        """
        # Ensure data is on GPU
        if raw_csi.device != self.device:
            raw_csi = raw_csi.to(self.device, non_blocking=True)

        # 1. Log-Magnitude Feature
        # |H| -> log10(|H|), Shape: (B, Q, T, N, M)
        mag = raw_csi.abs()
        log_mag = torch.log10(mag + self.eps)

        # 2. Differential Phase Feature (SFO Removal)
        # Calculate diff along Subcarrier axis (dim=-1) to remove linear SFO
        phase = raw_csi.angle()
        diff_phase = torch.diff(phase, dim=-1, prepend=phase[..., 0:1])

        # 3. Stack & Flatten
        # Stack features: (B, Q, T, N, M, 2)
        features_stacked = torch.stack([log_mag, diff_phase], dim=-1)

        # Permute to move M (Subcarriers) to the end: (B, Q, T, N, 2, M)
        features_permuted = features_stacked.permute(0, 1, 2, 3, 5, 4)

        # Flatten Q, T, N, 2 into 'Channels'
        B = features_permuted.shape[0]
        M_sub = features_permuted.shape[-1]
        features_flat = features_permuted.reshape(B, -1, M_sub)

        # 4. Instance Normalization
        # Normalize per instance, per channel to focus on shape
        return F.instance_norm(features_flat)