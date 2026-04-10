# core/models/csi_encoder.py

import torch
import torch.nn as nn
from typing import List


class ResidualTCNBlock(nn.Module):
    """
    Residual temporal block for TCN.
    Input / Output shape:
        [B, C, T]
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        out = out + residual
        out = self.final_act(out)
        return out


class CSIEncoder(nn.Module):
    """
    Lightweight CSI encoder for AP-level reliability representation.

    Input:
        x: [B_flat, T, C, M]
            B_flat = batch_size * num_aps
            T = number of time steps
            C = feature channels
                - channel 0: normalized amplitude
                - channel 1..: wrapped phase-difference channels
            M = number of subcarriers

    Output:
        z: [B_flat, T, latent_dim]
    """
    def __init__(
        self,
        num_feature_channels: int,
        num_subcarriers: int,
        cnn_channels: List[int] = [16, 32],
        cnn_kernel_size: int = 3,
        tcn_hidden: int = 64,
        tcn_kernel_size: int = 3,
        tcn_dilations: List[int] = [1, 2, 4],
        mlp_hidden: int = 64,
        latent_dim: int = 128,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_feature_channels = num_feature_channels
        self.num_subcarriers = num_subcarriers
        self.latent_dim = latent_dim

        # --------------------------------------------------
        # 1) Per-time-step local encoder: 1D CNN over subcarriers
        # Input at each time step: [C, M]
        # Output at each time step: [cnn_channels[-1]]
        # --------------------------------------------------
        cnn_layers = []
        in_ch = num_feature_channels

        for out_ch in cnn_channels:
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=cnn_kernel_size,
                    padding=cnn_kernel_size // 2,
                )
            )
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.local_cnn = nn.Sequential(*cnn_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, C, M] -> [B, C, 1]

        local_feature_dim = cnn_channels[-1]

        # --------------------------------------------------
        # 2) Temporal projection before TCN
        # --------------------------------------------------
        self.temporal_in = nn.Sequential(
            nn.Linear(local_feature_dim, tcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------
        # 3) TCN over time
        # Input / output: [B_flat, tcn_hidden, T]
        # --------------------------------------------------
        tcn_blocks = []
        for dilation in tcn_dilations:
            tcn_blocks.append(
                ResidualTCNBlock(
                    channels=tcn_hidden,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*tcn_blocks)

        # --------------------------------------------------
        # 4) Latent head
        # --------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(tcn_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, latent_dim),
        )

        # --------------------------------------------------
        # 5) Optional projection head
        # --------------------------------------------------
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B_flat, T, C, M]
            return_projection: whether to return projected features

        Returns:
            z: [B_flat, T, latent_dim] or [B_flat, T, proj_dim]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B_flat, T, C, M], got {tuple(x.shape)}")

        b_flat, t_steps, c_channels, m_subcarriers = x.shape

        if c_channels != self.num_feature_channels:
            raise ValueError(
                f"Expected num_feature_channels={self.num_feature_channels}, got {c_channels}"
            )
        if m_subcarriers != self.num_subcarriers:
            raise ValueError(
                f"Expected num_subcarriers={self.num_subcarriers}, got {m_subcarriers}"
            )

        # --------------------------------------------------
        # Per-time-step local encoding
        # [B_flat, T, C, M] -> [B_flat*T, C, M]
        # --------------------------------------------------
        x_local = x.reshape(b_flat * t_steps, c_channels, m_subcarriers)

        local_feat = self.local_cnn(x_local)              # [B_flat*T, C_last, M]
        local_feat = self.global_pool(local_feat)         # [B_flat*T, C_last, 1]
        local_feat = local_feat.squeeze(-1)               # [B_flat*T, C_last]

        # Restore temporal dimension
        local_feat = local_feat.view(b_flat, t_steps, -1) # [B_flat, T, local_feature_dim]

        # --------------------------------------------------
        # Temporal projection
        # --------------------------------------------------
        temporal_feat = self.temporal_in(local_feat)      # [B_flat, T, tcn_hidden]

        # --------------------------------------------------
        # TCN expects [B, C, T]
        # --------------------------------------------------
        temporal_feat = temporal_feat.transpose(1, 2).contiguous()  # [B_flat, tcn_hidden, T]
        temporal_feat = self.tcn(temporal_feat)                     # [B_flat, tcn_hidden, T]
        temporal_feat = temporal_feat.transpose(1, 2).contiguous()  # [B_flat, T, tcn_hidden]

        # --------------------------------------------------
        # Latent mapping
        # --------------------------------------------------
        z = self.mlp(temporal_feat)  # [B_flat, T, latent_dim]

        if return_projection:
            return self.projection(z)  # [B_flat, T, proj_dim]

        return z