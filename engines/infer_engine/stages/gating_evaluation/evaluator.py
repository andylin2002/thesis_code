# engines/infer_engine/stages/gating_evaluation/evaluator.py

import os
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from core.models.csi2vec_mlp import CSI2VecMLP
from engines.adapt_engine.stages.represent.stage import RepresentStage

class GatingEvaluator:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Hyperparameters
        self.embed_dim = config.get('EMBED_DIM', 16)
        self.temp_scale = config.get('GATING_TEMP', 1.0)
        
        self.represent_stage = RepresentStage(config, device)
        self.model = self._build_model()

    def _build_model(self) -> torch.nn.Module:
        """ Load the CSI2Vec MLP model and its weights """
        csi_info = self.config.get('CSI_DIMENSIONS', {})
        n_ant = csi_info.get('NUM_RX_ANTENNAS', 3)
        n_taps = self.config.get('C_MAX_TAPS', 16)
        
        input_dim = n_ant * n_taps
        model = CSI2VecMLP(input_dim=input_dim, embedding_dim=self.embed_dim).to(self.device)
        
        # Load weights from checkpoint 
        ckpt_path = self.config.get('CHECKPOINT_PATH', 'checkpoint/Office.ckpt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            state = ckpt.get('state_dict', ckpt)
            # Remove lightning prefix 'model.'
            clean_state = {k.replace('model.', ''): v for k, v in state.items()}
            model.load_state_dict(clean_state, strict=False)
        
        model.eval()
        return model

    @torch.no_grad()
    def evaluate(self, raw_csi_block: torch.Tensor, mmp_gain: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Main interface to get gating weights (Strictly vectorized) [cite: 21] """
        if raw_csi_block.dim() == 4:
            raw_csi_block = raw_csi_block.unsqueeze(0)
            
        _, Q, T, _, _ = raw_csi_block.shape
        
        # 1. Map CSI to Topological Tensor 'Z' (Q, T, D)
        representation = self.represent_stage.process(raw_csi_block)
        topological_Z = self.model(representation).view(Q, T, -1)
        
        # 2. Extract Physical Gain 'g' (Q, T) [cite: 8, 9]
        if mmp_gain is not None:
            physical_gain = mmp_gain.squeeze(0) if mmp_gain.dim() == 3 else mmp_gain
        else:
            raise ValueError(
                "[GatingEvaluator]"
                "Raw CSI amplitude suffers from small-scale fading (destructive interference). "
                "You must provide the resolved physical gain from the MMP algorithm to serve as a reliable physical prior mask."
            )

        # 3. Compute Neuro-Symbolic Gating Weights [cite: 11, 12]
        epd_gating, tpd_gating = self._get_gating_logic(topological_Z, physical_gain)
        
        return epd_gating, tpd_gating

    def _get_gating_logic(self, topological_Z: torch.Tensor, physical_gain: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Vectorized pipeline for geometric and physical analysis [cite: 22] """
        Q, T, D = topological_Z.shape
        eps = 1e-6
        
        # --- Phase 1: Physical Gain Preprocessing ---
        # Temporal smoothing: max over {t-1, t, t+1} 
        gain_pad = F.pad(physical_gain.unsqueeze(1), (1, 1), mode='replicate')
        smooth_gain = F.max_pool1d(gain_pad, kernel_size=3, stride=1).squeeze(1)
        
        # Min-Max Scaling per-AP to [0, 1] 
        g_min = smooth_gain.min(dim=1, keepdim=True).values
        g_max = smooth_gain.max(dim=1, keepdim=True).values
        norm_gain = (smooth_gain - g_min) / (g_max - g_min + eps)

        # --- Phase 2: Geometric Topological Analysis  ---
        # 1. TPD: Continuity via Cosine Similarity 
        # Calculate step vectors V_t = Z_t - Z_{t-1}
        step_vectors = topological_Z[:, 1:, :] - topological_Z[:, :-1, :] # (Q, T-1, D)
        # Pad to align V_t with V_{t+1} while maintaining shape (Q, T-1)
        v_padded = F.pad(step_vectors.permute(0, 2, 1), (0, 1), mode='replicate').permute(0, 2, 1)
        v_curr = step_vectors
        v_next = v_padded[:, 1:, :]
        cos_sim = F.cosine_similarity(v_curr, v_next, dim=-1) # (Q, T-1)
        
        # 2. EPD: Trajectory Tortuosity over window W=5 
        # Pad Z for sliding window: (Q, T+4, D)
        z_pad = F.pad(topological_Z.permute(0, 2, 1), (2, 2), mode='replicate').permute(0, 2, 1)
        # Numerator: Sum of adjacent step lengths ||Z_i+1 - Z_i||
        lengths = torch.norm(z_pad[:, 1:, :] - z_pad[:, :-1, :], dim=-1) # (Q, T+3)
        path_len = F.avg_pool1d(lengths.unsqueeze(1), kernel_size=4, stride=1).squeeze(1) * 4
        # Denominator: Net displacement ||Z_{t+2} - Z_{t-2}||
        net_disp = torch.norm(z_pad[:, 4:, :] - z_pad[:, :-4, :], dim=-1)
        tortuosity = path_len / (net_disp + eps) # (Q, T)

        # --- Phase 3: Neuro-Symbolic Joint Gating ---
        # 1. TPD Gating: Gain-masked directional consistency
        tpd_gating = norm_gain[:, :-1] * F.relu(cos_sim)
        
        # 2. EPD Gating: Gain-masked Softmax consensus
        # Per-AP scale-invariant standardization (MAD) 
        median = tortuosity.median(dim=1, keepdim=True).values
        mad = (tortuosity - median).abs().median(dim=1, keepdim=True).values + eps
        std_tortuosity = (tortuosity - median) / mad
        
        # Temperature-scaled Softmax across Q APs with Gain prior
        exp_term = norm_gain * torch.exp(-std_tortuosity / self.temp_scale)
        epd_gating = exp_term / (exp_term.sum(dim=0, keepdim=True) + eps)
        
        return epd_gating, tpd_gating