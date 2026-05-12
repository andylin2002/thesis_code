# core/models/encoder.py

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .blocks import SubcarrierEncoder, TemporalEncoder, APContextEncoder


class CSIManifoldEncoder(nn.Module):
    """Pattern -> shared latent [B,Q,T,D]."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.num_aps = len(config["ACCESS_POINTS"])
        self.num_rx_antennas = int(config["CSI_DIMENSIONS"]["NUM_RX_ANTENNAS"])
        self.num_subcarriers = int(config["CSI_DIMENSIONS"]["NUM_SUBCARRIERS"])

        if self.num_rx_antennas < 2:
            raise ValueError("NUM_RX_ANTENNAS must be at least 2")

        self.num_phase_diff_channels = (
            self.num_rx_antennas * (self.num_rx_antennas - 1) // 2
        )
        self.num_pattern_channels = 1 + self.num_phase_diff_channels

        conv_channels: List[int] = list(config.get("NEURAL_CNN_CHANNELS", [16, 32]))
        conv_kernel_size = int(config.get("NEURAL_CNN_KERNEL_SIZE", 3))
        temporal_dim = int(config.get("NEURAL_TEMPORAL_DIM", 64))
        tcn_kernel_size = int(config.get("NEURAL_TCN_KERNEL_SIZE", 3))
        tcn_dilations: List[int] = list(config.get("NEURAL_TCN_DILATIONS", [1, 2, 4]))
        num_heads = int(config.get("NEURAL_NUM_HEADS", 4))
        ap_num_layers = int(config.get("NEURAL_AP_NUM_LAYERS", 2))
        ap_ff_hidden = int(config.get("NEURAL_AP_FF_HIDDEN", 128))
        dropout = float(config.get("NEURAL_DROPOUT", 0.1))

        if temporal_dim % num_heads != 0:
            raise ValueError("NEURAL_TEMPORAL_DIM must be divisible by NEURAL_NUM_HEADS")

        self.subcarrier_encoder = SubcarrierEncoder(
            in_channels=self.num_pattern_channels,
            conv_channels=conv_channels,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalEncoder(
            in_dim=self.subcarrier_encoder.out_dim,
            hidden_dim=temporal_dim,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            dropout=dropout,
        )

        # Encode fixed AP identity.
        self.ap_embedding = nn.Embedding(self.num_aps, temporal_dim)

        self.ap_context_encoder = APContextEncoder(
            dim=temporal_dim,
            num_heads=num_heads,
            num_layers=ap_num_layers,
            ff_hidden_dim=ap_ff_hidden,
            dropout=dropout,
        )

        self.fuse = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.out_dim = temporal_dim

    def _check_input(self, pattern: torch.Tensor) -> None:
        if pattern.ndim != 5:
            raise ValueError(
                f"Expected pattern shape [B,Q,T,C,M], got {tuple(pattern.shape)}"
            )

        _, q, _, c, m = pattern.shape

        if q != self.num_aps:
            raise ValueError(f"Expected num_aps={self.num_aps}, got {q}")
        if c != self.num_pattern_channels:
            raise ValueError(
                f"Expected num_pattern_channels={self.num_pattern_channels}, got {c}"
            )
        if m != self.num_subcarriers:
            raise ValueError(
                f"Expected num_subcarriers={self.num_subcarriers}, got {m}"
            )

    def forward(self, pattern: torch.Tensor) -> torch.Tensor:
        self._check_input(pattern)

        b, q, t, c, m = pattern.shape

        x = pattern.reshape(b * q * t, c, m)
        x = self.subcarrier_encoder(x)       # [B*Q*T, D0]
        x = x.view(b * q, t, -1)             # [B*Q, T, D0]

        temporal = self.temporal_encoder(x)  # [B*Q, T, D]
        temporal_bqtd = temporal.view(b, q, t, -1)

        ap_ids = torch.arange(q, device=pattern.device)
        ap_emb = self.ap_embedding(ap_ids).view(1, q, 1, -1)
        temporal_bqtd = temporal_bqtd + ap_emb

        ctx = temporal_bqtd.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(b * t, q, -1)
        ctx = self.ap_context_encoder(ctx)
        ctx = ctx.view(b, t, q, -1).permute(0, 2, 1, 3).contiguous()

        fused = torch.cat([temporal_bqtd, ctx], dim=-1)
        fused = self.fuse(fused)

        return fused  # [B,Q,T,D]