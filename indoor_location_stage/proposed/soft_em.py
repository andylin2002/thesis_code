import torch
import numpy as np
from typing import Dict, Any, Optional

from . import soft_em_utils
from .._common import grid_tools
from .._common import math_tools

# Type definition
TypePropParams = Dict[str, torch.Tensor]

class SoftEM_Algorithm:
    """
    Parameter Optimizer for the Proposed Method.
    Uses Soft Assignments (Forward-Backward) to iteratively refine parameters.
    """
    def __init__(
            self, 
            features: torch.Tensor, 
            config: Dict[str, Any], 
            reference_grid: torch.Tensor,
            ap_data_info: Dict[str, Any],
            device: torch.device
        ):
        
        self.features = features
        self.config = config
        self.reference_grid = reference_grid
        self.device = device
        
        # Unpack AP info
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.ap_locations = ap_data_info['locations']
        self.ap_orientations = ap_data_info['orientations']
        
        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']
        
        self.G = reference_grid.shape[0]
        self.propagation_params: Optional[TypePropParams] = None
        self.MEPLL_PropParams = -torch.inf
        
        # Pre-calculate neighbor matrix for Forward-Backward
        G_index = torch.arange(self.G).to(device)
        self.neighbor_matrix = grid_tools.get_all_neighbor_indices(config, G_index, device)

        # --- Pre-calculate Static Geometry (Lookup Tables) ---
        # Angle Matrix (Q, G)
        self.grid_angle_qg = soft_em_utils.calculate_grid_angle_qg(
            reference_grid, self.ap_locations, self.ap_orientations
        ).to(device)

        # Distance Matrix (Q, G)
        self.grid_delay_qg = soft_em_utils.calculate_grid_delay_qg(
            reference_grid, self.ap_locations
        ).to(device)

        self.final_emission_log_probs: Optional[torch.Tensor] = None

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
                self.grid_delay_qg
            )
            self.final_emission_log_probs = emission_log_probs
            
            # 2. E-Step: Compute Spatio-Temporal Probability Distribution (Posterior)
            gamma_gt = soft_em_utils.run_forward_backward(
                emission_log_probs,
                self.neighbor_matrix,
                self.device
            )
            
            # 3. M-Step: Update Parameters using Soft Weights
            new_params = soft_em_utils.update_soft_parameters(
                self.features,
                self.propagation_params,
                gamma_gt,
                self.grid_angle_qg, 
                self.grid_delay_qg
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
    
    def get_final_epd(self) -> torch.Tensor:
        """ Returns the EPD from the last iteration. """
        if self.final_emission_log_probs is None:
            raise RuntimeError("Run step_parameters() first!")
        return self.final_emission_log_probs