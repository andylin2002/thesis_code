# engines/adapt_engine/stages/train/stage.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Dict

class TrainStage:
    """Computes InfoNCE loss and updates the injected encoder."""

    def __init__(self, config: Dict[str, Any], device: torch.device, encoder: torch.nn.Module):
        self.device = device
        self.tau = float(config.get("CONTRASTIVE_TAU", 0.07))
        self.Q = len(config["ACCESS_POINTS"])
        
        # Injected dependencies
        self.encoder = encoder
        self.optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=float(config.get("LR", 5e-4))
        )

    def step(self, z_proj: torch.Tensor) -> Dict[str, float]:
        """Performs one training step using projected latent vectors."""
        self.encoder.train()
        
        B, Q, D = z_proj.shape
        total_loss = 0.0
        pair_count = 0

        # Compute Contrastive Loss independently for each AP
        for q in range(self.Q):
            z_q = z_proj[:, q, :]
            z_q = F.normalize(z_q, dim=-1)

            # Positive pairs: temporally adjacent windows
            anchors = z_q[:-1]
            positives = z_q[1:]

            # Similarity matrix vs all other times in the batch
            sim_matrix = torch.matmul(anchors, z_q.T) / self.tau
            labels = torch.arange(anchors.shape[0], device=self.device) + 1
            
            loss = F.cross_entropy(sim_matrix, labels)
            total_loss += loss
            pair_count += 1

        total_loss = total_loss / pair_count

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"loss": total_loss.item()}

    def get_state_dict(self) -> Dict[str, Any]:
        """Exports encoder parameters."""
        return {k: v.cpu() for k, v in self.encoder.state_dict().items()}