# engines/neural_engine/stages/represent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple
from core.models.csi_encoder import CSIEncoder

class CrossAPAttention(nn.Module):
    """Evaluates parallel AP features to output Viterbi emission weights."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: [B, T, Q, D]
        scores = self.attention_head(z_q)  # [B, T, Q, 1]
        return F.softmax(scores.squeeze(-1), dim=-1)  # [B, T, Q]


class RepresentStage:
    """Preprocesses CSI and extracts spatial-temporal latents and attention weights."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.Q = len(config["ACCESS_POINTS"])
        self.N = int(config['CSI_DIMENSIONS']['NUM_RX_ANTENNAS'])
        self.M = int(config['CSI_DIMENSIONS']['NUM_SUBCARRIERS'])
        self.D = int(config.get("LATENT_DIM", 128))

        # 1. Shared feature extractor (Preserves T dimension)
        self.encoder = CSIEncoder(
            num_antennas=self.N, 
            num_subcarriers=self.M,
            latent_dim=self.D
        ).to(self.device)

        # 2. Cross-AP Attention module
        self.attention = CrossAPAttention(feature_dim=self.D).to(self.device)

    def process(self, raw_csi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            raw_csi: [B, Q, T, N, M]
        Returns:
            encoded: [B, T, Q, D] Latent features for temporal prediction
            viterbi_weights: [B, T, Q] Attention weights for emission probability
        """
        if raw_csi.device != self.device:
            raw_csi = raw_csi.to(self.device, non_blocking=True)

        B, Q, T, N, M = raw_csi.shape

        # --- 1. Preprocessing ---
        amplitude = torch.abs(raw_csi)
        amplitude = amplitude / (amplitude.mean(dim=(-1, -2), keepdim=True) + 1e-8)
        
        phase = torch.angle(raw_csi)
        phase_diff = phase[:, :, :, 1:, :] - phase[:, :, :, :-1, :]
        amplitude = amplitude[:, :, :, 1:, :]

        features = torch.stack([amplitude, phase_diff], dim=-1)
        
        # --- 2. Parallel AP Encoding ---
        # Group Batch and AP, keep Time (T) intact for the GRU
        features = features.reshape(B * Q, T, N - 1, M, 2)
        encoded = self.encoder(features, return_projection=False)  # [B*Q, T, D]
        
        # Reshape to explicit spatial-temporal format
        encoded = encoded.view(B, Q, T, -1).permute(0, 2, 1, 3)  # [B, T, Q, D]

        # --- 3. Compute Viterbi Emission Weights ---
        viterbi_weights = self.attention(encoded)  # [B, T, Q]

        return encoded, viterbi_weights