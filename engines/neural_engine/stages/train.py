# engines/neural_engine/stages/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Dict

class TemporalPredictor(nn.Module):
    """Predicts the latent state of t+1 given t."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.transition(z_t)


class TrainStage:
    """Trains the network using Uncertainty-Weighted Predictive Loss."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device, encoder: nn.Module, attention: nn.Module):
        self.device = device
        self.D = int(config.get("LATENT_DIM", 128))
        self.entropy_lambda = float(config.get("ENTROPY_LAMBDA", 0.1))
        
        # Modules to optimize
        self.encoder = encoder
        self.attention = attention
        self.predictor = TemporalPredictor(self.D).to(self.device)

        # Joint optimization
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.attention.parameters()},
            {'params': self.predictor.parameters()}
        ], lr=float(config.get("LR", 5e-4)))

    def step(self, z_features: torch.Tensor, viterbi_weights: torch.Tensor) -> Dict[str, float]:
        """
        Args:
            z_features: [B, T, Q, D]
            viterbi_weights: [B, T, Q]
        """
        self.encoder.train()
        self.attention.train()
        self.predictor.train()
        
        # 1. Temporal Prediction Setup (predict t+1 from t)
        z_t = z_features[:, :-1, :, :]          # [B, T-1, Q, D]
        z_t_next = z_features[:, 1:, :, :]      # [B, T-1, Q, D] Target
        weights_t = viterbi_weights[:, :-1, :]  # [B, T-1, Q] Weights at time t

        # Predict next state independently per AP
        z_t_pred = self.predictor(z_t)          # [B, T-1, Q, D]

        # 2. Prediction Error (Unreduced MSE per AP)
        mse_error = F.mse_loss(z_t_pred, z_t_next.detach(), reduction='none').mean(dim=-1)  # [B, T-1, Q]

        # 3. Uncertainty-Weighted Loss (Forces low weight on unpredictable APs)
        weighted_pred_loss = (mse_error * weights_t).sum(dim=-1).mean() 

        # 4. Entropy Regularization (Prevents weight collapse to a single AP)
        entropy = -torch.sum(weights_t * torch.log(weights_t + 1e-8), dim=-1).mean()
        
        # Total Loss
        total_loss = weighted_pred_loss - (self.entropy_lambda * entropy)

        # 5. Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "pred_loss": weighted_pred_loss.item(),
            "entropy": entropy.item()
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Exports all module parameters for the Symbolic Engine to consume."""
        return {
            "encoder": {k: v.cpu() for k, v in self.encoder.state_dict().items()},
            "attention": {k: v.cpu() for k, v in self.attention.state_dict().items()},
            "predictor": {k: v.cpu() for k, v in self.predictor.state_dict().items()}
        }