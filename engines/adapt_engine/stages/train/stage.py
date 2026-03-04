# engines/adapt_engine/stages/train/stage.py

from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from core.models.csi2vec_mlp import CSI2VecMLP

class TrainStage:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.lr = float(config.get("LR", 5e-4))
        self.embed_dim = int(config.get("EMBED_DIM", 16))
        self.margin = float(config.get("TRIPLET_MARGIN", 2.0))
        self.consensus_weight = float(config.get("CONSENSUS_WEIGHT", 1.0)) # 提升權重以增強 AP 共識
        
        self.Q = len(config['ACCESS_POINTS']) # 8 APs
        self.T = int(config.get('NUM_SAMPLE', 20)) # TAF Time steps
        
        self.model = CSI2VecMLP(
            input_dim=int(config.get("N_ANTENNAS", 3)) * int(config.get("C_MAX_TAPS", 16)),
            embedding_dim=self.embed_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def step(self, features: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        v_flat = self.model(features)
        B = v_flat.shape[0] // (self.Q * self.T)
        v = v_flat.view(B, self.Q, self.T, -1) # (B, Q, T, D)

        # 1. Temporal Triplet Loss with Safety Mask
        # Goal: Prevent pushing away neighbors (t vs t+2) which causes low TW/CT
        anchors = v[:, :, :-1, :].reshape(-1, self.embed_dim)
        positives = v[:, :, 1:, :].reshape(-1, self.embed_dim)
        
        # --- Time-window Masked Mining ---
        # Instead of random shuffle, we pick negatives that are at least 5 steps away
        # This prevents "Manifold Tearing"
        batch_size_flat = anchors.size(0)
        with torch.no_grad():
            # Generate indices that are guaranteed to be far in time
            shift = torch.randint(5, batch_size_flat - 5, (batch_size_flat,), device=self.device)
            neg_idx = (torch.arange(batch_size_flat, device=self.device) + shift) % batch_size_flat
        negatives = anchors[neg_idx]
        
        loss_temporal = self.triplet_loss(anchors, positives, negatives)

        # 2. AP Consensus Loss (The foundation for your Gating logic)
        # Goal: Force all 8 APs to agree on the same coordinate at time t
        v_centroid = v.mean(dim=1, keepdim=True) # (B, 1, T, D)
        loss_consensus = F.mse_loss(v, v_centroid.expand_as(v))

        # 3. Total Optimization
        total_loss = loss_temporal + self.consensus_weight * loss_consensus
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 4. Diagnostic Metrics: Help you identify if APs are actually converging
        with torch.no_grad():
            # Average distance between APs at the same time step
            ap_dispersion = torch.mean(torch.norm(v - v_centroid, p=2, dim=-1))

        return {
            "loss": total_loss.item(),
            "t_loss": loss_temporal.item(),        # Monitoring path continuity
            "c_loss": loss_consensus.item(),       # Monitoring AP consensus
            "ap_dispersion": ap_dispersion.item()  # The lower, the more LoS-like consensus
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {k: v.cpu() for k, v in self.model.state_dict().items()}