# engines/infer_engine/stages/result_estimation/proposed/soft_em.py

import torch
import numpy as np
from typing import Dict, Any, Optional

from . import soft_em_utils

# Type definition
TypePropParams = Dict[str, torch.Tensor]

class SoftEMAlgorithm:
    """
    Parameter Optimizer for the Proposed Method.
    Uses Soft Assignments (Forward-Backward) to iteratively refine parameters.
    """
    def __init__(
            self, 
            features: torch.Tensor, 
            config: Dict[str, Any], 
            reference_grid: torch.Tensor,
            ap_data: Dict[str, Any],
            device: torch.device
        ):
        
        self.features = features
        self.config = config
        self.reference_grid = reference_grid
        self.ap_data = ap_data
        self.device = device
        
        # Unpack AP info
        self.ap_locations = ap_data['locations']
        self.ap_orientations = ap_data['orientations']
        self.grid_angle_qg = ap_data['grid_angle_qg']
        self.grid_delay_qg = ap_data['grid_delay_qg']
        
        self.num_ap = len(config['ACCESS_POINTS'])
        self.num_sample = config['NUM_SAMPLE']
        
        self.propagation_params: Optional[TypePropParams] = None
        self.MEPLL_PropParams = -torch.inf
        
        # Pre-calculate neighbor matrix for Forward-Backward
        nm = ap_data['neighbor_matrix']
        if not isinstance(nm, torch.Tensor):
            nm = torch.as_tensor(nm)
        self.neighbor_matrix = nm.to(device=device, dtype=torch.long)

        self.final_emission_log_probs: Optional[torch.Tensor] = None
        self.final_spatiotemporal_probs: Optional[torch.Tensor] = None
        self.emission_gating: Optional[torch.Tensor] = None

    def initialize_params(self) -> TypePropParams:
        """
        Initialize propagation parameters(weight, bias, offset)
        """
        
        self.propagation_params = soft_em_utils.initialize_parameters(self.config, self.device)
        
        return self.propagation_params

    def step_parameters(self):
        """
        Execute the Soft EM Loop (Baum-Welch style) with Convergence Check.
        
        1. E-Step: Calculate EPD.
        2. E-Step: Calculate Posterior (Gamma).
        3. M-Step: Update parameters.
        """
        if self.propagation_params is None:
            self.initialize_params()
            
        max_iter = self.config.get('EM_MAX_ITER', 10)
        
        for i in range(max_iter):
            # Backup old params for convergence check
            old_params = {k: v.clone() for k, v in self.propagation_params.items()}

            # 1. E-Step: Calculate Emission Probability Distribution
            emission_log_probs = soft_em_utils.calculate_emission_log_probs(
                self.features,
                self.propagation_params,
                self.grid_angle_qg, 
                emission_gating=self.emission_gating
            )
            self.final_emission_log_probs = emission_log_probs
            
            # 2. E-Step: Compute Spatio-Temporal Probability Distribution (Posterior)
            stpd_gt = soft_em_utils.run_forward_backward(
                emission_log_probs,
                self.neighbor_matrix,
                self.device
            )
            self.final_spatiotemporal_probs = stpd_gt
            
            # 3. M-Step: Update Parameters using Soft Weights
            new_params = soft_em_utils.update_soft_parameters(
                self.features,
                self.propagation_params,
                stpd_gt,
                self.grid_angle_qg
            )
            
            self.propagation_params = new_params
            
            # 4. Check Convergence
            if self._check_parameter_convergence(old_params, new_params):
                break

    def _check_parameter_convergence(
        self, 
        old_params: TypePropParams, 
        new_params: TypePropParams
    ) -> bool:
        """
        Check if max parameter change is below tolerance.
        """
        max_diff = 0.0
        for key in old_params:
            diff = torch.abs(new_params[key] - old_params[key]).max().item()
            if diff > max_diff:
                max_diff = diff
        
        return max_diff < 1e-4
    
    def set_emission_gating(self, emission_gating: Optional[torch.Tensor]) -> None:
        """
        emission_gating: (Q, T) in [0,1] or None
        - None => disable gating (pure physics / equal voting)
        """
        if emission_gating is None:
            self.emission_gating = None
            return

        if not isinstance(emission_gating, torch.Tensor):
            raise TypeError("emission_gating must be a torch.Tensor or None")

        Q, T = self.features.size(0), self.features.size(1)
        if emission_gating.shape != (Q, T):
            raise ValueError(
                f"emission_gating shape mismatch: expected (Q,T)=({Q},{T}), got {tuple(emission_gating.shape)}"
            )

        eg = emission_gating.to(device=self.device, dtype=torch.float32)

        eg = eg.clamp(0.0, 1.0)

        self.emission_gating = eg

    def get_final_epd(self) -> torch.Tensor:
        """ Returns the EPD from the last iteration. """
        if self.final_emission_log_probs is None:
            raise RuntimeError("Run step_parameters() first!")
        return self.final_emission_log_probs
    
    def get_final_stpd(self) -> torch.Tensor:
        """ Returns the STPD from the last iteration. """
        if self.final_spatiotemporal_probs is None:
            raise RuntimeError("Run step_parameters() first!")
        return self.final_spatiotemporal_probs