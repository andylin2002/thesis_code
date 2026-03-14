# engines/neural_engine/stages/represent.py

import torch
from typing import Any, Dict
from core.models.csi_encoder import CSIEncoder

class RepresentStage:
    """Preprocesses CSI and extracts latent features using a shared encoder."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.Q = len(config["ACCESS_POINTS"])
        self.N = int(config.get("N_ANTENNAS", 3))
        self.M = int(config.get("N_SUBCARRIERS", 21))
        self.T = int(config.get("NUM_SAMPLE", 20))

        # Shared encoder (AP-independent)
        self.encoder = CSIEncoder().to(self.device)

    def process(self, raw_csi: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        """Transforms raw CSI into de-geometrized latent/projection vectors."""
        if raw_csi.device != self.device:
            raw_csi = raw_csi.to(self.device, non_blocking=True)

        B, Q, T, N, M = raw_csi.shape

        # 1. Amplitude normalization
        amplitude = torch.abs(raw_csi)
        amplitude = amplitude / (amplitude.mean(dim=(-1, -2), keepdim=True) + 1e-8)

        # 2. Phase difference extraction
        phase = torch.angle(raw_csi)
        phase_diff = phase[:, :, :, 1:, :] - phase[:, :, :, :-1, :]

        # 3. Align amplitude with phase difference (drop the reference antenna)
        amplitude = amplitude[:, :, :, 1:, :]

        # 4. Stack features -> [B, Q, T, N-1, M, 2]
        features = torch.stack([amplitude, phase_diff], dim=-1)

        # 5. Parallel AP encoding
        features = features.reshape(B * Q, T, N - 1, M, 2)
        encoded = self.encoder(features, return_projection=return_projection)

        # 6. Restore AP dimension -> [B, Q, D]
        encoded = encoded.view(B, Q, -1)

        return encoded