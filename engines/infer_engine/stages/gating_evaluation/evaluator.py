# engines/infer_engine/stages/gating_evaluation/evaluator.py

import os
import torch
from typing import Tuple, Optional

from core.models.csi_encoder import CSIEncoder
from engines.adapt_engine.stages.represent.stage import RepresentStage

class GatingEvaluator:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Hyperparameters
        self.sigma_z = config.get('GATING_SIGMA_Z', 1.0)
        self.sigma_v = config.get('GATING_SIGMA_V', 0.5)
        self.embedding_dim = config.get('EMBEDDING_DIM', 16)
        
        self.represent_stage = RepresentStage(config, device)
        self.model = self._build_and_load_model()

    def _build_and_load_model(self) -> torch.nn.Module:
        # Input channels = N_ANTENNAS * 2 (LogMag + PhaseDiff)
        n_ant = self.config['CSI_DIMENSIONS']['NUM_RX_ANTENNAS']
        in_channels = n_ant * 2 
        
        model = CSIEncoder(in_channels=in_channels, embedding_dim=self.embedding_dim).to(self.device)
        model.eval()

        ckpt_path = self.config.get('CHECKPOINT_PATH', 'checkpoint/Office.ckpt')
        if os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                # Remove Lightning prefix
                new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"[GatingEvaluator] Error loading weights: {e}")
        
        return model

    @torch.no_grad()
    def evaluate(self, raw_csi: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Ensure batch dim
        if raw_csi.dim() == 4:
            raw_csi = raw_csi.unsqueeze(0)
            
        # Features: (B*Q*T, Channels, Subcarriers)
        features = self.represent_stage.process(raw_csi)
        
        # Inference: (B*Q*T, D)
        z_flat = self.model(features) 
        
        # Reshape to (Q, T, D) for topology calc
        _, Q, T, _, _ = raw_csi.shape
        Z = z_flat.view(Q, T, -1)

        epd_gating = self._compute_epd_gating(Z)
        tpd_gating = self._compute_tpd_gating(Z)
        
        return epd_gating, tpd_gating

    def _compute_epd_gating(self, Z: torch.Tensor) -> torch.Tensor:
        """ Calculates spatial density score (EPD) """
        Q, T, D = Z.shape
        weights = torch.zeros(Q, T, device=self.device)

        for t in range(T):
            t_start, t_end = max(0, t-1), min(T, t+2)
            current = Z[:, t, :].unsqueeze(1)
            neighbors = Z[:, t_start:t_end, :].reshape(-1, D).unsqueeze(0)
            
            # RBF Kernel
            dists = torch.sum((current - neighbors) ** 2, dim=2)
            scores = torch.exp(-dists / (self.sigma_z ** 2)).sum(dim=1)
            weights[:, t] = scores

        norm = weights.sum(dim=0, keepdim=True) + 1e-6
        return weights / norm

    def _compute_tpd_gating(self, Z: torch.Tensor) -> torch.Tensor:
        """ Calculates velocity consistency score (TPD) """
        Q, T, D = Z.shape
        if T < 3: 
            return torch.ones(Q, T-1, device=self.device) / Q

        # Central Difference Velocity
        V = torch.zeros(Q, T, D, device=self.device)
        V[:, 1:-1, :] = (Z[:, 2:, :] - Z[:, :-2, :]) / 2.0
        V[:, 0, :]    = Z[:, 1, :] - Z[:, 0, :]
        V[:, -1, :]   = Z[:, -1, :] - Z[:, -2, :]
        
        weights = torch.zeros(Q, T-1, device=self.device)
        for t in range(T - 1):
            current_v = V[:, t, :].unsqueeze(1)
            t_start, t_end = max(0, t-1), min(T, t+2)
            neighbors_v = V[:, t_start:t_end, :].reshape(-1, D).unsqueeze(0)
            
            dists = torch.sum((current_v - neighbors_v) ** 2, dim=2)
            scores = torch.exp(-dists / (self.sigma_v ** 2)).sum(dim=1)
            weights[:, t] = scores

        norm = weights.sum(dim=0, keepdim=True) + 1e-6
        return weights / norm