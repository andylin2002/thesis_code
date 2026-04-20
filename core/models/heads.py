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

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, latent_bqtd: torch.Tensor) -> torch.Tensor:
        if latent_bqtd.ndim != 4:
            raise ValueError(
                f"Expected latent shape [B,Q,T,D], got {tuple(latent_bqtd.shape)}"
            )

        pooled = latent_bqtd.mean(dim=(1, 2))   # [B, D]
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

        # Per-AP feature fusion
        self.ap_token_proj = nn.Sequential(
            nn.Linear(latent_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Joint AP mixing at each time step
        self.ap_mixer = nn.Sequential(
            nn.Linear(num_aps, num_aps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_aps, num_aps),
        )

        # Cross-AP refinement in hidden space
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
            raise ValueError(
                f"Expected num_aps={self.num_aps}, got {q}"
            )
        if d != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {d}"
            )

        # Global block context
        block_ctx = self.block_proj(embedding_be).view(b, 1, 1, d).expand(-1, q, t, -1)

        # Time-wise AP context
        time_ctx = latent_bqtd.mean(dim=2, keepdim=True).expand(-1, -1, t, -1)
        latent_centered = latent_bqtd - time_ctx

        # Stronger AP competition statistics
        ap_mean = latent_bqtd.mean(dim=1, keepdim=True).expand(-1, q, -1, -1)
        ap_max = latent_bqtd.max(dim=1, keepdim=True).values.expand(-1, q, -1, -1)

        feat = torch.cat(
            [
                latent_bqtd,
                latent_centered,
                time_ctx,
                block_ctx,
                latent_bqtd - ap_mean,
                latent_bqtd - ap_max,
            ],
            dim=-1,
        )  # [B,Q,T,6D]

        token = self.ap_token_proj(feat)  # [B,Q,T,H]

        # Joint ranking per time step
        token_btqh = token.permute(0, 2, 1, 3).contiguous()   # [B,T,Q,H]
        bt = b * t
        token_2d = token_btqh.view(bt, q, self.hidden_dim)    # [B*T,Q,H]

        # Mix APs jointly across the AP dimension
        mixed = token_2d.transpose(1, 2).contiguous()         # [B*T,H,Q]
        mixed = self.ap_mixer(mixed)                          # [B*T,H,Q]
        mixed = mixed.transpose(1, 2).contiguous()            # [B*T,Q,H]

        # Residual refinement with global AP-set summary
        ap_set_summary = mixed.mean(dim=1, keepdim=True)      # [B*T,1,H]
        refined = mixed + self.cross_ap_refine(mixed - ap_set_summary)

        logits_btq = self.score_head(refined).squeeze(-1)     # [B*T,Q]
        logits_btq = logits_btq.view(b, t, q).contiguous()    # [B,T,Q]

        return logits_btq