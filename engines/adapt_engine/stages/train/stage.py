# engines/adapt_engine/stages/train/stage.py

from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from core.models.csi2vec_mlp import CSI2VecMLP

class TrainStage:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.lr = float(config.get("LR", 1e-3))
        self.embed_dim = int(config.get("EMBED_DIM", 16))
        self.margin = float(config.get("TRIPLET_MARGIN", 10.0))
        
        self.Q = len(config['ACCESS_POINTS'])
        self.T = int(config.get('NUM_SAMPLE', 20))
        
        # Dynamic input dimension calculation
        n_ant = int(config.get("N_ANTENNAS", 3))
        c_max = int(config.get("C_MAX_TAPS", 16))
        input_dim = n_ant * c_max

        self.model = CSI2VecMLP(input_dim=input_dim, embedding_dim=self.embed_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.TripletMarginLoss(margin=self.margin, p=2)

    def step(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Input: (B*Q*T, D)
        """
        self.model.train()
        
        # 1. Extract vector embeddings
        v_flat = self.model(features)
        
        # 2. Restore trajectory structure
        B = v_flat.shape[0] // (self.Q * self.T)
        v = v_flat.view(B, self.Q, self.T, -1)

        # 3. Temporal Triplet Mining (t vs t+1)
        anchor = v[:, :, :-1, :].reshape(-1, self.embed_dim)
        positive = v[:, :, 1:, :].reshape(-1, self.embed_dim)
        
        # 4. Global Shuffled Negatives
        idx = torch.randperm(anchor.size(0))
        negative = anchor[idx]

        # 5. Optimize Triplet Margin Loss
        loss = self.criterion(anchor, positive, negative)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def get_state_dict(self) -> Dict[str, Any]:
        return {k: v.cpu() for k, v in self.model.state_dict().items()}