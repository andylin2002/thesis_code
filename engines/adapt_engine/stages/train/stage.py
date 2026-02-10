# engines/adapt_engine/stages/train/stage.py

from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from engines.adapt_engine.models.csi_encoder import CSIEncoder


class TrainStage:
    """
    TrainStage (The Coach): Orchestrates model training using Contrastive Learning.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        
        # Hyperparameters
        self.learning_rate = float(config.get("ADAPT_LR", 1e-3))
        self.embedding_dim = int(config.get("ADAPT_EMBED_DIM", 16))
        self.margin = float(config.get("ADAPT_TRIPLET_MARGIN", 1.0))

        # Input config
        Q = len(config['ACCESS_POINTS']) 
        T = int(config.get('NUM_SAMPLE', 20))
        N = int(config.get("N_ANTENNAS", 3))
        
        # Calculate flat input channels (Q * T * N * 2 features)
        self.in_channels = Q * T * N * 2

        # Initialize Model (ResNet-1D)
        self.model = CSIEncoder(
            in_channels=self.in_channels, 
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss Function: dist(anchor, pos) < dist(anchor, neg) - margin
        self.criterion = nn.TripletMarginLoss(margin=self.margin, p=2)

    def step(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.
        Input features shape: (B, Channels, M)
        """
        self.model.train()
        
        # 1. Forward Pass
        embeddings = self.model(features) # (B, Embedding_Dim)
        
        # 2. Temporal Triplet Mining
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return {}

        # Anchor: t
        anchor = embeddings[0 : -1]
        
        # Positive: t + 1 (Next time step)
        positive = embeddings[1 : ]
        
        # Negative: Random roll (Likely distant in time)
        shift = batch_size // 2
        negative = torch.roll(embeddings, shifts=shift, dims=0)[0 : -1]

        # 3. Calculate Loss
        loss = self.criterion(anchor, positive, negative)

        # 4. Backward & Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns model weights on CPU for safe IPC."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
