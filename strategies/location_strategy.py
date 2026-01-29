# strategies/location_strategy.py

import torch
from typing import Optional
from core.interfaces import ILocationEstimator
from indoor_location_stage._common import grid_tools

from indoor_location_stage.baseline.estimator import BaselineEstimator
from indoor_location_stage.proposed.estimator import ProposedEstimator

class BaselineLocationStrategy(ILocationEstimator):
    """
    Strategy for Baseline localization (Hard EM / Viterbi).
    Directly wraps the legacy function.
    """
    def __init__(self, config, device, reference_grid, directions_vectors):
        self.config = config
        self.device = device
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors

        # Pre-calculate AP info ONCE during initialization
        self.num_ap = len(config['ACCESS_POINTS'])
        self.ap_data_info = {
            'locations': grid_tools.get_ap_locations(config, self.num_ap, device),
            'orientations': torch.tensor(
                [data.get('ORIENTATION_DEG', 0) for data in config['ACCESS_POINTS'].values()], 
                dtype=torch.float32, 
                device=device
            )
        }

    def estimate(self, features: torch.Tensor) -> torch.Tensor:
        estimator = BaselineEstimator(
            features=features,
            config=self.config,
            reference_grid=self.reference_grid,
            ap_data=self.ap_data_info, 
            device=self.device
        )
        
        trajectory = estimator.solve()
            
        return trajectory

class ProposedLocationStrategy(ILocationEstimator):
    """
    Strategy for Proposed localization (Physics-Aware + optional MLP gating)
    """
    def __init__(self, config, device, reference_grid, directions_vectors):
        self.config = config
        self.device = device
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors

        self.gating_model: Optional[torch.nn.Module] = None
        self._pending_gating_state_dict: Optional[dict] = None

        # Pre-calculate AP info ONCE during initialization
        self.num_ap = len(config['ACCESS_POINTS'])
        self.ap_data_info = {
            'locations': grid_tools.get_ap_locations(config, self.num_ap, device),
            'orientations': torch.tensor(
                [data.get('ORIENTATION_DEG', 0) for data in config['ACCESS_POINTS'].values()], 
                dtype=torch.float32, 
                device=device
            )
        }
    
    # =========================================================
    # Public API for INFER_Worker
    # =========================================================
    def set_gating_state_dict(self, state_dict: Optional[dict]) -> None:
        """
        INFER_Worker will call this when it receives a new model update from queues['model'].
        - state_dict is stored and will be applied lazily.
        - if state_dict is None: disable gating.
        """
        self._pending_gating_state_dict = state_dict
        if state_dict is None:
            self.gating_model = None

    # =========================================================
    # Main estimation
    # =========================================================
    def estimate(self, features: torch.Tensor) -> torch.Tensor:
        """
        Executes the Physics-Aware pipeline
        """
        self._ensure_gating_model_ready()

        # Get gating weight
        ap_time_gate = None
        if self.gating_model is not None:
            ap_time_gate = self._inference_ap_time_gate(features)
        
        # Instantiate the Proposed Estimator
        estimator = ProposedEstimator(
            features=features,
            config=self.config,
            reference_grid=self.reference_grid,
            ap_data=self.ap_data_info,
            ap_time_gate=ap_time_gate, 
            device=self.device,
        )

        trajectory = estimator.solve()
            
        return trajectory

    # =========================================================
    # Internal: lazy build/load gating model
    # =========================================================
    def _ensure_gating_model_ready(self) -> None:
        """
        Lazy init/load:
        - If no pending state_dict -> do nothing (pure physics path)
        - If pending state_dict exists -> build model (if needed) + load + eval
        """
        if self._pending_gating_state_dict is None:
            return

        state_dict = self._pending_gating_state_dict
        self._pending_gating_state_dict = None  # consume once

        if self.gating_model is None:
            self.gating_model = self._build_gating_model().to(self.device)

        self.gating_model.load_state_dict(state_dict)
        self.gating_model.eval()

    def _build_gating_model(self) -> torch.nn.Module:
        """
        IMPORTANT:
        You haven't provided the MLP model definition yet.
        Keep this as a hook. When you implement mlp_worker, define the same model here.
        """
        raise NotImplementedError(
            "[ProposedLocationStrategy] gating model is not implemented yet. "
            "Pure physics path is OK. If you start sending state_dict, "
            "please implement _build_gating_model() with the correct architecture."
        )

    # =========================================================
    # Inference
    # =========================================================
    @torch.no_grad()
    def _inference_ap_time_gate(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (Q,T,C,5)
        returns:  (Q,T) in [0,1]
        """
        if self.gating_model is None:
            raise RuntimeError("gating_model is None, but _inference_ap_time_gate() was called.")

        gate_features = self._build_gate_features(features)  # (Q,T,F)
        model_device = next(self.gating_model.parameters()).device
        gate_features = gate_features.to(model_device)

        ap_time_gate = self.gating_model(gate_features)  # (Q,T) or (Q,T,1)
        if ap_time_gate.dim() == 3 and ap_time_gate.size(-1) == 1:
            ap_time_gate = ap_time_gate.squeeze(-1)

        Q, T, _, _ = features.shape
        if ap_time_gate.shape != (Q, T):
            raise ValueError(
                f"ap_time_gate shape mismatch: expected (Q,T)=({Q},{T}), got {tuple(ap_time_gate.shape)}"
            )

        return torch.clamp(ap_time_gate, 0.0, 1.0)

    def _build_gate_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Build MLP input features from physics features.
        features: (Q,T,C,5)
        output:   (Q,T,F)
        """
        aoa_sprd = features[..., 1]   # (Q,T,C)
        tof_sprd = features[..., 3]   # (Q,T,C)
        gain = features[..., 4]       # (Q,T,C)

        aoa_sprd_min = aoa_sprd.min(dim=2).values
        aoa_sprd_mean = aoa_sprd.mean(dim=2)

        tof_sprd_min = tof_sprd.min(dim=2).values
        tof_sprd_mean = tof_sprd.mean(dim=2)

        gain_sum = gain.sum(dim=2) + 1e-9
        gain_norm = gain / gain_sum.unsqueeze(-1)
        gain_top1 = gain_norm.max(dim=2).values

        return torch.stack(
            [aoa_sprd_min, aoa_sprd_mean, tof_sprd_min, tof_sprd_mean, gain_top1],
            dim=-1,
        )

