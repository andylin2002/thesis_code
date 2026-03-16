# engines/symbolic_engine/stages/gating_evaluation/evaluator.py

import torch
from typing import Dict, Any

from core.models.csi_encoder import CSIEncoder
from engines.neural_engine.stages.represent import CrossAPAttention

class GatingEvaluator:
    """
    Real-time inference client. 
    Receives updated weights from the Neural Engine and outputs Viterbi emission weights.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        self.N = int(config['CSI_DIMENSIONS']['NUM_RX_ANTENNAS'])
        self.M = int(config['CSI_DIMENSIONS']['NUM_SUBCARRIERS'])
        self.D = int(config.get("LATENT_DIM", 128))

        # 1. Instantiate only the required inference modules
        self.encoder = CSIEncoder(
            num_antennas=self.N, 
            num_subcarriers=self.M,
            latent_dim=self.D
        ).to(self.device)
        
        self.attention = CrossAPAttention(feature_dim=self.D).to(self.device)

        # 2. Lock models in evaluation mode (disables dropout/batchnorm updates)
        self.encoder.eval()
        self.attention.eval()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Hot-swaps model weights dynamically from the queue payload."""
        try:
            # Extract the specific sub-module weights
            self.encoder.load_state_dict(state_dict["encoder"])
            self.attention.load_state_dict(state_dict["attention"])
            print("[GatingEvaluator] AI weights successfully hot-swapped.")
        except Exception as e:
            print(f"[GatingEvaluator] Failed to load state dict: {e}")

    @torch.no_grad()
    def evaluate(self, raw_csi_block: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_csi_block: [B, Q, T, N, M]
        Returns:
            viterbi_weights: [B, T, Q] (Softmax probabilities summing to 1 over Q)
        """
        if raw_csi_block.device != self.device:
            raw_csi_block = raw_csi_block.to(self.device)

        is_single_block = (raw_csi_block.dim() == 4)
        if is_single_block:
            raw_csi_block = raw_csi_block.unsqueeze(0)

        B, Q, T, N, M = raw_csi_block.shape

        # --- 1. Lightweight Preprocessing ---
        amplitude = torch.abs(raw_csi_block)
        amplitude = amplitude / (amplitude.mean(dim=(-1, -2), keepdim=True) + 1e-8)
        
        phase = torch.angle(raw_csi_block)
        phase_diff = phase[:, :, :, 1:, :] - phase[:, :, :, :-1, :]
        amplitude = amplitude[:, :, :, 1:, :]

        features = torch.stack([amplitude, phase_diff], dim=-1)
        
        # --- 2. Forward Pass ---
        features = features.reshape(B * Q, T, N - 1, M, 2)
        encoded = self.encoder(features, return_projection=False)  # [B*Q, T, D]
        
        # Reshape to explicit spatial-temporal format
        encoded = encoded.view(B, Q, T, -1).permute(0, 2, 1, 3)  # [B, T, Q, D]

        # --- 3. Compute Weights ---
        viterbi_weights = self.attention(encoded)  # [B, T, Q]

        # --- 4. Output ---
        if is_single_block:
            return viterbi_weights[0].transpose(0, 1)
        else:
            return viterbi_weights.transpose(1, 2)