# engines/symbolic_engine/stages/gating_evaluation/evaluator.py

import os
import torch
import torch.nn.functional as F
from typing import Optional

from core.models.csi_encoder import CSIEncoder
from engines.neural_engine.stages.represent import RepresentStage

class GatingEvaluator:
    """
    Evaluates LoS/NLoS reliability using Temporal Cosine Similarity.
    Applies Competitive Softmax and scales by Q, so weights sum to Q across APs.
    """
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        self.represent_stage = RepresentStage(config, device)
        self.model = self._build_model()

    def _build_model(self) -> torch.nn.Module:
        n_ant = int(self.config.get('N_ANTENNAS', 3))
        n_sub = int(self.config.get('N_SUBCARRIERS', 21))
        
        model = CSIEncoder(num_antennas=n_ant, num_subcarriers=n_sub).to(self.device)
        
        scene_name = self.config.get("SCENARIO_NAME", "default_scene")
        ckpt_path = os.path.join("checkpoint", f"{scene_name}.ckpt")
        
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt.get('model_state', ckpt))
        else:
            print(f"[GatingEvaluator] Warning: No checkpoint at {ckpt_path}.")
            
        model.eval()
        return model

    @torch.no_grad()
    def evaluate(self, raw_csi_block: torch.Tensor) -> torch.Tensor:
        """
        Output: gating_weights [B, Q, T] where sum(gating_weights, dim=1) == Q
        """
        is_single_block = (raw_csi_block.dim() == 4)
        if is_single_block:
            raw_csi_block = raw_csi_block.unsqueeze(0)
            
        B, Q, T, N, M = raw_csi_block.shape
        
        # 1. Flatten temporal dimension -> [B*T, Q, 1, N, M]
        flat_csi = raw_csi_block.transpose(1, 2).reshape(B * T, Q, 1, N, M)
        
        # 2. Map to Projection Space
        frame_Z = self.represent_stage.process(flat_csi, return_projection=True)
        
        # 3. Restore temporal dimension -> [B, Q, T, D]
        D = frame_Z.shape[-1]
        frame_Z = frame_Z.view(B, T, Q, D).transpose(1, 2)
        
        # 4. Compute Competitive Gating Weights
        gating_weights = self._compute_competitive_softmax(frame_Z)

        if is_single_block:
            gating_weights = gating_weights.squeeze(0)
        
        return gating_weights

    def _compute_competitive_softmax(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Calculates Temporal Cosine Similarity, applies Softmax, and scales by Q.
        """
        B, Q, T, D = Z.shape
        
        # 1. Temporal Cosine Similarity (Z_t compared to Z_{t-1})
        Z_prev = torch.cat([Z[:, :, 0:1, :], Z[:, :, :-1, :]], dim=2)
        cos_sim = F.cosine_similarity(Z, Z_prev, dim=-1) # [B, Q, T]
        
        # 2. Temperature scaling to sharpen the differences
        # Make sure GATING_TEMP in config.yaml is small (e.g., 0.01)
        tau = float(self.config.get('GATING_TEMP', 0.01))
        scaled_sim = cos_sim / tau
        
        # 3. Competitive Allocation: Softmax across the AP dimension (dim=1)
        softmax_weights = F.softmax(scaled_sim, dim=1)
        
        # 4. Scale by Q to preserve the overall Emission Log-Likelihood budget
        gating_weights = softmax_weights * Q
        
        return gating_weights