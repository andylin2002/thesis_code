# core/models/reliability_model.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


class ResidualTCNBlock(nn.Module):
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


class APContextBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ff_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop1(attn_out)

        ff_in = self.norm2(x)
        ff_out = self.ff(ff_in)
        x = x + self.drop2(ff_out)
        return x


class CSIEncoder(nn.Module):
    def __init__(
        self,
        num_feature_channels: int,
        num_subcarriers: int,
        num_aps: int,
        cnn_channels: List[int],
        cnn_kernel_size: int,
        tcn_hidden: int,
        tcn_kernel_size: int,
        tcn_dilations: List[int],
        ap_num_heads: int,
        ap_num_layers: int,
        ap_ff_hidden: int,
        mlp_hidden: int,
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.num_feature_channels = num_feature_channels
        self.num_subcarriers = num_subcarriers
        self.num_aps = num_aps
        self.latent_dim = latent_dim

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
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        local_feature_dim = cnn_channels[-1]

        self.temporal_in = nn.Sequential(
            nn.Linear(local_feature_dim, tcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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

        self.ap_context = nn.ModuleList(
            [
                APContextBlock(
                    embed_dim=tcn_hidden,
                    num_heads=ap_num_heads,
                    ff_hidden=ap_ff_hidden,
                    dropout=dropout,
                )
                for _ in range(ap_num_layers)
            ]
        )

        self.fusion = nn.Sequential(
            nn.Linear(tcn_hidden * 2, tcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(tcn_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B_flat,T,C,M], got {tuple(x.shape)}")

        b_flat, t_steps, c_channels, m_subcarriers = x.shape

        if c_channels != self.num_feature_channels:
            raise ValueError(
                f"Expected num_feature_channels={self.num_feature_channels}, got {c_channels}"
            )
        if m_subcarriers != self.num_subcarriers:
            raise ValueError(
                f"Expected num_subcarriers={self.num_subcarriers}, got {m_subcarriers}"
            )
        if b_flat % self.num_aps != 0:
            raise ValueError(
                f"B_flat={b_flat} is not divisible by num_aps={self.num_aps}"
            )

        batch_size = b_flat // self.num_aps

        x_local = x.reshape(b_flat * t_steps, c_channels, m_subcarriers)
        local_feat = self.local_cnn(x_local)
        local_feat = self.global_pool(local_feat).squeeze(-1)
        local_feat = local_feat.view(b_flat, t_steps, -1)

        temporal_feat = self.temporal_in(local_feat)
        temporal_feat = temporal_feat.transpose(1, 2).contiguous()
        temporal_feat = self.tcn(temporal_feat)
        temporal_feat = temporal_feat.transpose(1, 2).contiguous()

        h = temporal_feat.view(batch_size, self.num_aps, t_steps, -1)
        h = h.permute(0, 2, 1, 3).contiguous()
        h = h.view(batch_size * t_steps, self.num_aps, -1)

        for block in self.ap_context:
            h = block(h)

        h = h.view(batch_size, t_steps, self.num_aps, -1)
        h = h.permute(0, 2, 1, 3).contiguous()

        base = temporal_feat.view(batch_size, self.num_aps, t_steps, -1)
        fused = torch.cat([base, h], dim=-1)
        fused = self.fusion(fused)

        fused = fused.view(b_flat, t_steps, -1)
        z = self.mlp(fused)
        return z


class ScoreHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected head input shape [B_flat,T,D], got {tuple(x.shape)}"
            )

        logits = self.mlp(x).squeeze(-1)  # [B_flat, T]
        return logits


class NeuralReliabilityModel(nn.Module):
    """Pattern -> AP score logits."""

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

        self.latent_dim = int(config.get("LATENT_DIM", 128))
        self.num_heads = int(config.get("NEURAL_NUM_HEADS", 4))

        if self.latent_dim % self.num_heads != 0:
            raise ValueError("LATENT_DIM must be divisible by NEURAL_NUM_HEADS")

        cnn_channels = list(config.get("NEURAL_CNN_CHANNELS", [16, 32]))
        cnn_kernel_size = int(config.get("NEURAL_CNN_KERNEL_SIZE", 3))
        tcn_hidden = int(config.get("NEURAL_TCN_HIDDEN", 64))
        tcn_kernel_size = int(config.get("NEURAL_TCN_KERNEL_SIZE", 3))
        tcn_dilations = list(config.get("NEURAL_TCN_DILATIONS", [1, 2, 4]))
        ap_num_layers = int(config.get("NEURAL_AP_NUM_LAYERS", 2))
        ap_ff_hidden = int(config.get("NEURAL_AP_FF_HIDDEN", 128))
        mlp_hidden = int(config.get("NEURAL_MLP_HIDDEN", 64))
        dropout = float(config.get("NEURAL_DROPOUT", 0.1))

        self.encoder = CSIEncoder(
            num_feature_channels=self.num_pattern_channels,
            num_subcarriers=self.num_subcarriers,
            num_aps=self.num_aps,
            cnn_channels=cnn_channels,
            cnn_kernel_size=cnn_kernel_size,
            tcn_hidden=tcn_hidden,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilations=tcn_dilations,
            ap_num_heads=self.num_heads,
            ap_num_layers=ap_num_layers,
            ap_ff_hidden=ap_ff_hidden,
            mlp_hidden=mlp_hidden,
            latent_dim=self.latent_dim,
            dropout=dropout,
        )
        self.head = ScoreHead(self.latent_dim)

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

    def forward_logits(self, pattern: torch.Tensor) -> torch.Tensor:
        self._check_input(pattern)

        b, q, t, c, m = pattern.shape
        x = pattern.reshape(b * q, t, c, m)      # [B*Q, T, C, M]
        z = self.encoder(x)                      # [B*Q, T, D]
        logits_bq_t = self.head(z)              # [B*Q, T]
        logits_btq = logits_bq_t.reshape(b, q, t).permute(0, 2, 1).contiguous()  # [B, T, Q]
        return logits_btq

    def logits_to_reliability(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits shape [B,T,Q], got {tuple(logits.shape)}"
            )

        if logits.shape[-1] != self.num_aps:
            raise ValueError(
                f"Expected last dim num_aps={self.num_aps}, got {logits.shape[-1]}"
            )

        return self.num_aps * torch.softmax(logits, dim=-1)

    def forward(
        self,
        pattern: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_logits(pattern)
        reliability = self.logits_to_reliability(logits)

        if return_logits:
            return reliability, logits
        return reliability