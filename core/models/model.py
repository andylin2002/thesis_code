# core/models/model.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .encoder import CSIManifoldEncoder
from .heads import BlockEmbeddingHead, ReliabilityHead


class NeuralReliabilityModel(nn.Module):
    """Pattern -> embedding + reliability logits."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.num_aps = len(config["ACCESS_POINTS"])

        self.encoder = CSIManifoldEncoder(config)
        self.embedding_head = BlockEmbeddingHead(self.encoder.out_dim, config)
        self.reliability_head = ReliabilityHead(self.encoder.out_dim, config)

    def encode(self, pattern: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(pattern)                    # [B,Q,T,D]
        embedding = self.embedding_head(latent)          # [B,E]
        return embedding

    def forward_features(self, pattern: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encoder(pattern)                   # [B,Q,T,D]
        embedding = self.embedding_head(latent)          # [B,E]
        logits = self.reliability_head(latent, embedding)  # [B,T,Q]

        return {
            "latent": latent,
            "embedding": embedding,
            "logits": logits,
        }

    def forward_logits(self, pattern: torch.Tensor) -> torch.Tensor:
        return self.forward_features(pattern)["logits"]

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
        return_embedding: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        out = self.forward_features(pattern)
        reliability = self.logits_to_reliability(out["logits"])

        if return_logits and return_embedding:
            return reliability, out["logits"], out["embedding"]

        if return_logits:
            return reliability, out["logits"]

        if return_embedding:
            return reliability, out["embedding"]

        return reliability