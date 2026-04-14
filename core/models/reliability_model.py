# core/models/reliability_model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple

from core.models.csi_encoder import CSIEncoder


class CrossAPReliabilityHead(nn.Module):
    """Map latent features to AP-wise reliability."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(feature_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        encoded: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if encoded.ndim != 4:
            raise ValueError(
                f"Expected encoded with shape [B, T, Q, D], got {tuple(encoded.shape)}"
            )

        _, _, num_aps, _ = encoded.shape

        x = self.norm(encoded)                  # [B, T, Q, D]
        logits = self.score_mlp(x).squeeze(-1) # [B, T, Q]
        reliability = num_aps * F.softmax(logits, dim=-1)

        if return_logits:
            return reliability, logits
        return reliability


class NeuralReliabilityModel(nn.Module):
    """Full model: pattern -> reliability."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.num_aps = len(config["ACCESS_POINTS"])
        self.num_rx_antennas = int(config["CSI_DIMENSIONS"]["NUM_RX_ANTENNAS"])
        self.num_subcarriers = int(config["CSI_DIMENSIONS"]["NUM_SUBCARRIERS"])
        self.latent_dim = int(config.get("LATENT_DIM", 128))

        if self.num_rx_antennas < 2:
            raise ValueError("NUM_RX_ANTENNAS must be at least 2")

        self.num_phase_diff_channels = self.num_rx_antennas * (self.num_rx_antennas - 1) // 2
        self.num_pattern_channels = 1 + self.num_phase_diff_channels

        self.encoder = CSIEncoder(
            num_feature_channels=self.num_pattern_channels,
            num_subcarriers=self.num_subcarriers,
            num_aps=self.num_aps,
            cnn_channels=config.get("NEURAL_CNN_CHANNELS", [16, 32]),
            cnn_kernel_size=int(config.get("NEURAL_CNN_KERNEL_SIZE", 3)),
            tcn_hidden=int(config.get("NEURAL_TCN_HIDDEN", 64)),
            tcn_kernel_size=int(config.get("NEURAL_TCN_KERNEL_SIZE", 3)),
            tcn_dilations=config.get("NEURAL_TCN_DILATIONS", [1, 2, 4]),
            ap_num_heads=int(config.get("NEURAL_AP_NUM_HEADS", 4)),
            ap_num_layers=int(config.get("NEURAL_AP_NUM_LAYERS", 2)),
            ap_ff_hidden=int(config.get("NEURAL_AP_FF_HIDDEN", 128)),
            mlp_hidden=int(config.get("NEURAL_MLP_HIDDEN", 64)),
            latent_dim=self.latent_dim,
            proj_dim=int(config.get("PROJ_DIM", 64)),
            dropout=float(config.get("NEURAL_DROPOUT", 0.1)),
        )

        self.reliability_head = CrossAPReliabilityHead(
            feature_dim=self.latent_dim,
            hidden_dim=int(config.get("RELIABILITY_HEAD_HIDDEN", 64)),
            dropout=float(config.get("RELIABILITY_HEAD_DROPOUT", 0.1)),
        )

    def _encode_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        if pattern.ndim != 5:
            raise ValueError(
                f"Expected pattern with shape [B, Q, T, C, M], got {tuple(pattern.shape)}"
            )

        batch_size, num_aps, num_steps, num_channels, num_subcarriers = pattern.shape

        if num_aps != self.num_aps:
            raise ValueError(f"Expected num_aps={self.num_aps}, got {num_aps}")
        if num_channels != self.num_pattern_channels:
            raise ValueError(
                f"Expected num_pattern_channels={self.num_pattern_channels}, got {num_channels}"
            )
        if num_subcarriers != self.num_subcarriers:
            raise ValueError(
                f"Expected num_subcarriers={self.num_subcarriers}, got {num_subcarriers}"
            )

        encoder_input = pattern.view(
            batch_size * num_aps,
            num_steps,
            num_channels,
            num_subcarriers,
        )  # [B*Q, T, C, M]

        encoded_per_ap = self.encoder(
            encoder_input,
            return_projection=False,
        )  # [B*Q, T, D]

        encoded = encoded_per_ap.view(
            batch_size,
            num_aps,
            num_steps,
            -1,
        ).permute(0, 2, 1, 3).contiguous()  # [B, T, Q, D]

        return encoded

    def forward(
        self,
        pattern: torch.Tensor,
        return_logits: bool = False,
        return_latent: bool = False,
    ):
        encoded = self._encode_pattern(pattern)  # [B, T, Q, D]

        reliability, logits = self.reliability_head(
            encoded,
            return_logits=True,
        )  # [B, T, Q], [B, T, Q]

        if return_logits and return_latent:
            return reliability, logits, encoded
        if return_logits:
            return reliability, logits
        if return_latent:
            return reliability, encoded
        return reliability