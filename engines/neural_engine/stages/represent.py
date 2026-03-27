# engines/neural_engine/stages/represent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union

from core.models.csi_encoder import CSIEncoder


class CrossAPAttention(nn.Module):
    """
    Context-aware AP reliability scorer.

    Input:
        z_q: [B, T, Q, D]

    Output:
        probs : [B, T, Q]
        scores: [B, T, Q]  (optional, before softmax)

    Design:
        For each AP at each time step, score it using:
            - its own latent feature z_q
            - cross-AP context mean(z_q over Q)
            - relative deviation from context: z_q - context

        This is still lightweight, but it is now genuinely relative across APs.
    """

    def __init__(self, feature_dim: int, temperature: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)

        in_dim = feature_dim * 3
        hidden_dim = max(feature_dim, 32)

        self.attention_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        z_q: torch.Tensor,
        return_scores: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            z_q: [B, T, Q, D]
            return_scores: whether to also return raw scores before softmax

        Returns:
            probs:  [B, T, Q]
            scores: [B, T, Q] (if return_scores=True)
        """
        if z_q.ndim != 4:
            raise ValueError(f"Expected z_q with shape [B, T, Q, D], got {tuple(z_q.shape)}")

        # Cross-AP context at each time step
        context = z_q.mean(dim=2, keepdim=True)      # [B, T, 1, D]
        context = context.expand_as(z_q)             # [B, T, Q, D]

        # Relative deviation from cross-AP context
        delta = z_q - context                        # [B, T, Q, D]

        # Concatenate self / context / relative info
        head_input = torch.cat([z_q, context, delta], dim=-1)  # [B, T, Q, 3D]

        # Raw scores before softmax
        scores = self.attention_head(head_input).squeeze(-1)   # [B, T, Q]

        # Relative AP probability
        probs = F.softmax(scores / self.temperature, dim=-1)   # [B, T, Q]

        if return_scores:
            return probs, scores
        return probs


class RepresentStage:
    """
    Preprocesses CSI and extracts spatial-temporal latents and AP probabilities.

    Pipeline:
        raw CSI [B,Q,T,N,M]
            -> amplitude normalization + inter-antenna phase difference
            -> reshape to [B*Q,T,N-1,M,2]
            -> shared CNN+GRU encoder
            -> encoded latent [B,T,Q,D]
            -> cross-AP attention probabilities [B,T,Q]
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.Q = len(config["ACCESS_POINTS"])
        self.N = int(config["CSI_DIMENSIONS"]["NUM_RX_ANTENNAS"])
        self.M = int(config["CSI_DIMENSIONS"]["NUM_SUBCARRIERS"])
        self.D = int(config.get("LATENT_DIM", 128))

        # Optional config for attention temperature
        self.attention_temperature = float(config.get("ATTENTION_TEMPERATURE", 0.5))

        # Shared CNN+GRU encoder
        self.encoder = CSIEncoder(
            num_antennas=self.N,
            num_subcarriers=self.M,
            latent_dim=self.D
        ).to(self.device)

        # Context-aware cross-AP scorer
        self.attention = CrossAPAttention(
            feature_dim=self.D,
            temperature=self.attention_temperature
        ).to(self.device)

    def process(
        self,
        raw_csi: torch.Tensor,
        return_scores: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Args:
            raw_csi: [B, Q, T, N, M]
            return_scores: whether to also return raw attention logits

        Returns:
            encoded : [B, T, Q, D]
            ap_probs: [B, T, Q]
            scores  : [B, T, Q] (optional)
        """
        if raw_csi.device != self.device:
            raw_csi = raw_csi.to(self.device, non_blocking=True)

        if raw_csi.ndim != 5:
            raise ValueError(f"Expected raw_csi with shape [B,Q,T,N,M], got {tuple(raw_csi.shape)}")

        B, Q, T, N, M = raw_csi.shape

        if Q != self.Q:
            raise ValueError(f"Configured Q={self.Q}, but input Q={Q}")
        if N != self.N:
            raise ValueError(f"Configured N={self.N}, but input N={N}")
        if M != self.M:
            raise ValueError(f"Configured M={self.M}, but input M={M}")

        if N < 2:
            raise ValueError("NUM_RX_ANTENNAS must be at least 2 to compute inter-antenna phase difference.")

        # ------------------------------------------------------------
        # 1. Preprocessing
        # ------------------------------------------------------------
        # Amplitude normalization
        amplitude = torch.abs(raw_csi)  # [B,Q,T,N,M]
        amplitude = amplitude / (amplitude.mean(dim=(-1, -2), keepdim=True) + 1e-8)

        # Keep antenna dimension aligned with phase difference (N-1)
        amplitude = amplitude[:, :, :, 1:, :]  # [B,Q,T,N-1,M]

        # Inter-antenna phase difference
        phase = torch.angle(raw_csi)  # [B,Q,T,N,M]
        phase_delta = phase[:, :, :, 1:, :] - phase[:, :, :, :-1, :]  # [B,Q,T,N-1,M]
        phase_diff = torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta))  # wrapped diff

        # Stack amplitude and phase_diff as 2-channel input
        features = torch.stack([amplitude, phase_diff], dim=-1)  # [B,Q,T,N-1,M,2]

        # ------------------------------------------------------------
        # 2. Shared AP encoding
        # ------------------------------------------------------------
        # Flatten AP into batch, preserve temporal dimension T
        features = features.reshape(B * Q, T, N - 1, M, 2)  # [B*Q,T,N-1,M,2]

        encoded = self.encoder(features, return_projection=False)  # [B*Q,T,D]

        # Restore explicit [B,T,Q,D]
        encoded = encoded.view(B, Q, T, -1).permute(0, 2, 1, 3).contiguous()  # [B,T,Q,D]

        # ------------------------------------------------------------
        # 3. Cross-AP reliability probability
        # ------------------------------------------------------------
        if return_scores:
            ap_probs, scores = self.attention(encoded, return_scores=True)  # [B,T,Q], [B,T,Q]
            return encoded, ap_probs, scores

        ap_probs = self.attention(encoded, return_scores=False)  # [B,T,Q]
        return encoded, ap_probs