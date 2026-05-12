# core/models/heads.py

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockEmbeddingHead(nn.Module):
    """[B,Q,T,D] -> [B,E]."""

    def __init__(self, in_dim: int, config: Dict[str, Any]):
        super().__init__()

        embed_dim = int(config.get("NEURAL_EMBED_DIM", 128))
        hidden_dim = int(config.get("NEURAL_EMBED_HIDDEN", in_dim))
        dropout = float(config.get("NEURAL_DROPOUT", 0.1))

        pooled_dim = in_dim * 3

        self.norm = nn.LayerNorm(pooled_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, latent_bqtd: torch.Tensor) -> torch.Tensor:
        if latent_bqtd.ndim != 4:
            raise ValueError(
                f"Expected latent shape [B,Q,T,D], got {tuple(latent_bqtd.shape)}"
            )

        b, q, t, d = latent_bqtd.shape
        flat = latent_bqtd.reshape(b, q * t, d)

        # Summarize the whole block with distribution statistics.
        pooled_mean = flat.mean(dim=1)
        pooled_max = flat.max(dim=1).values
        pooled_std = flat.std(dim=1, unbiased=False)

        pooled = torch.cat([pooled_mean, pooled_max, pooled_std], dim=-1)

        pooled = self.norm(pooled)
        z = self.mlp(pooled)
        z = F.normalize(z, dim=-1)

        return z  # [B,E]


class ReliabilityHead(nn.Module):
    """[B,Q,T,D] + [B,E] -> [B,T,Q]."""

    def __init__(self, latent_dim: int, config: Dict[str, Any]):
        super().__init__()

        embed_dim = int(config.get("NEURAL_EMBED_DIM", 128))
        hidden_dim = int(config.get("NEURAL_HEAD_HIDDEN", latent_dim))
        num_aps = len(config["ACCESS_POINTS"])
        dropout = float(config.get("NEURAL_DROPOUT", 0.1))

        if hidden_dim <= 0:
            raise ValueError("NEURAL_HEAD_HIDDEN must be > 0")

        self.num_aps = num_aps
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.block_proj = nn.Linear(embed_dim, latent_dim)

        # Fuse local, temporal, cross-AP, and block-level context.
        self.ap_token_proj = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.ap_mixer = nn.Sequential(
            nn.Linear(num_aps, num_aps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_aps, num_aps),
        )

        self.cross_ap_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latent_bqtd: torch.Tensor,
        embedding_be: torch.Tensor,
    ) -> torch.Tensor:
        if latent_bqtd.ndim != 4:
            raise ValueError(
                f"Expected latent shape [B,Q,T,D], got {tuple(latent_bqtd.shape)}"
            )
        if embedding_be.ndim != 2:
            raise ValueError(
                f"Expected embedding shape [B,E], got {tuple(embedding_be.shape)}"
            )

        b, q, t, d = latent_bqtd.shape

        if q != self.num_aps:
            raise ValueError(f"Expected num_aps={self.num_aps}, got {q}")
        if d != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {d}")

        block_ctx = self.block_proj(embedding_be).view(b, 1, 1, d)
        block_ctx = block_ctx.expand(-1, q, t, -1)

        ap_temporal_mean = latent_bqtd.mean(dim=2, keepdim=True)
        temporal_centered = latent_bqtd - ap_temporal_mean

        cross_ap_mean = latent_bqtd.mean(dim=1, keepdim=True)
        cross_ap_centered = latent_bqtd - cross_ap_mean

        feat = torch.cat(
            [
                latent_bqtd,
                temporal_centered,
                cross_ap_centered,
                block_ctx,
            ],
            dim=-1,
        )  # [B,Q,T,4D]

        token = self.ap_token_proj(feat)  # [B,Q,T,H]

        token_btqh = token.permute(0, 2, 1, 3).contiguous()
        token_2d = token_btqh.view(b * t, q, self.hidden_dim)

        mixed = token_2d.transpose(1, 2).contiguous()
        mixed = self.ap_mixer(mixed)
        mixed = mixed.transpose(1, 2).contiguous()

        ap_set_summary = mixed.mean(dim=1, keepdim=True)
        refined = mixed + self.cross_ap_refine(mixed - ap_set_summary)

        logits_btq = self.score_head(refined).squeeze(-1)
        logits_btq = logits_btq.view(b, t, q).contiguous()

        return logits_btq