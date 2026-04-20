# core/models/blocks.py

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
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
        self.norm1 = nn.GroupNorm(1, channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.drop2(y)

        return self.out_act(y + residual)


class SubcarrierEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: List[int],
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        if len(conv_channels) == 0:
            raise ValueError("conv_channels must not be empty")

        layers = []
        ch_in = in_channels

        for ch_out in conv_channels:
            layers.append(
                nn.Conv1d(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.GroupNorm(1, ch_out))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            ch_in = ch_out

        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = conv_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B_flat,C,M], got {tuple(x.shape)}"
            )

        y = self.net(x)
        y = self.pool(y).squeeze(-1)
        return y


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dilations: List[int],
        dropout: float,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        blocks = []
        for d in dilations:
            blocks.append(
                ResidualTCNBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B_flat,T,D_in], got {tuple(x.shape)}"
            )

        y = self.in_proj(x)
        y = y.transpose(1, 2).contiguous()
        y = self.tcn(y)
        y = y.transpose(1, 2).contiguous()
        return y


class APContextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B_flat,Q,D], got {tuple(x.shape)}"
            )

        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(y)

        y = self.norm2(x)
        y = self.ff(y)
        x = x + self.drop2(y)

        return x


class APContextEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ff_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                APContextBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x