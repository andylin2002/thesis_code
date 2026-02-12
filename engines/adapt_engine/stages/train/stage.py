# engines/adapt_engine/stages/train/stage.py

from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from core.models.csi_encoder import CSIEncoder


class TrainStage:
    """
    TrainStage (The Coach): Learns the Latent Topology.
    
    It uses a Shared Encoder approach:
    1. Encodes each AP/Time instance independently.
    2. Recovers the (B, Q, T) structure.
    3. Applies Temporal Triplet Loss to ensure trajectory smoothness per AP.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        
        # Hyperparameters
        self.learning_rate = float(config.get("ADAPT_LR", 1e-3))
        self.embedding_dim = int(config.get("ADAPT_EMBED_DIM", 16))
        self.margin = float(config.get("ADAPT_TRIPLET_MARGIN", 1.0))

        # Topology Config (Crucial for reshaping)
        self.Q = len(config['ACCESS_POINTS']) 
        self.T = int(config.get('NUM_SAMPLE', 20))
        
        # CRITICAL CHANGE: 
        # Since Q and T are moved to Batch dim, Channels only contain Antenna info.
        N = int(config.get("N_ANTENNAS", 3))
        self.in_channels = N * 2 

        # Initialize Model (Shared ResNet-1D)
        self.model = CSIEncoder(
            in_channels=self.in_channels, 
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss Function
        self.criterion = nn.TripletMarginLoss(margin=self.margin, p=2)

    def step(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Input features shape: (B*Q*T, N*2, M)
        """
        self.model.train()
        
        # 1. Forward Pass (Shared Encoder)
        # The model treats every AP and Timestamp as an independent sample.
        # embeddings_flat: (Total_Samples, D)
        embeddings_flat = self.model(features) 
        
        # 2. Recover Topology Structure
        # We need to reshape back to (B, Q, T, D) to perform temporal mining.
        total_samples = embeddings_flat.shape[0]
        
        # Dynamic Batch Size Calculation
        if total_samples % (self.Q * self.T) != 0:
            # Drop incomplete batch if necessary (safety check)
            return {}
            
        B = total_samples // (self.Q * self.T)
        
        # Reshape: (Batch, AP, Time, Dim)
        embeddings = embeddings_flat.view(B, self.Q, self.T, -1)

        # 3. Temporal Triplet Mining (Per AP)
        # Goal: Force z_t to be close to z_{t+1} for the SAME AP.
        
        # Anchor: Time 0 to T-2
        # Shape: (B, Q, T-1, D)
        anchor = embeddings[:, :, 0:-1, :]
        
        # Positive: Time 1 to T-1 (The next immediate step)
        # Shape: (B, Q, T-1, D)
        positive = embeddings[:, :, 1:, :]
        
        # Negative: Time Shuffling (Hard Negative)
        # We roll the time axis to pick a non-adjacent time step as negative.
        # This forces the model to distinguish "neighbors" from "distant times".
        shift = self.T // 2
        negative = torch.roll(embeddings, shifts=shift, dims=2)[:, :, 0:-1, :]

        # 4. Flatten for Loss Calculation
        # TripletLoss expects (N, D), so we merge B, Q, T again.
        loss = self.criterion(
            anchor.reshape(-1, self.embedding_dim),
            positive.reshape(-1, self.embedding_dim),
            negative.reshape(-1, self.embedding_dim)
        )

        # 5. Backward & Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns model weights on CPU for safe IPC."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}