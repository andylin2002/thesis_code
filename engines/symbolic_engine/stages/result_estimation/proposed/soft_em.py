# engines/symbolic_engine/stages/result_estimation/proposed/soft_em.py

import os
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

        self.emission_log_probs_qgt: Optional[torch.Tensor] = None
        self.posterior_gt: Optional[torch.Tensor] = None
        self.reliability: Optional[torch.Tensor] = None

        self.tof_params = self._calculate_tof_params()
        self.log_pi_qtc = soft_em_utils.calculate_log_pi_qtc(
            self.features,
            self.tof_params,
        )

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
            self.emission_log_probs_qgt = soft_em_utils.calculate_emission_log_probs(
                self.config, 
                self.features,
                self.propagation_params,
                self.grid_angle_qg, 
                self.log_pi_qtc
            )
            emission_log_probs_gt = torch.sum(self.emission_log_probs_qgt, dim=0)  # (G, T)
            
            # 2. E-Step: Compute Spatio-Temporal Probability Distribution (Posterior)
            posterior_gt = soft_em_utils.run_forward_backward(
                emission_log_probs_gt,
                self.neighbor_matrix,
                self.device
            )
            self.posterior_gt = posterior_gt
            
            # 3. M-Step: Update Parameters using Soft Weights
            new_params = soft_em_utils.update_soft_parameters(
                self.config,
                self.features,
                self.propagation_params,
                posterior_gt,
                self.grid_angle_qg, 
                self.log_pi_qtc
            )
            
            self.propagation_params = new_params
            
            # 4. Check Convergence
            if self._check_parameter_convergence(old_params, new_params):
                break

        self._recompute_current_state()

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
    
    def set_reliability(self, reliability: Optional[torch.Tensor]) -> None:
        if reliability is None:
            self.reliability = None
            return

        if not isinstance(reliability, torch.Tensor):
            reliability = torch.as_tensor(reliability, dtype=torch.float32, device=self.device)
        else:
            reliability = reliability.to(device=self.device, dtype=torch.float32)

        if reliability.shape != (self.num_ap, self.num_sample):
            raise ValueError(
                f"[SoftEM] reliability shape mismatch: "
                f"expected {(self.num_ap, self.num_sample)}, got {tuple(reliability.shape)}"
            )

        self.reliability = reliability

    def get_emission_log_probs_qgt(self) -> torch.Tensor:
        """
        Returns per-AP emission log probabilities with shape (Q, G, T).

        Args:
            apply_reliability:
                False -> return raw per-AP emission from symbolic model
                True  -> return reliability-weighted per-AP emission

        Returns:
            Tensor of shape (Q, G, T)
        """
        if self.emission_log_probs_qgt is None:
            raise RuntimeError("Run step_parameters() first!")

        emission_log_probs_qgt = self.emission_log_probs_qgt.clone()

        return emission_log_probs_qgt

    def get_emission_log_probs_gt(self) -> torch.Tensor:
        if self.emission_log_probs_qgt is None:
            raise RuntimeError("Run step_parameters() first!")

        emission_log_probs_qgt = self.emission_log_probs_qgt.clone()

        if self.reliability is not None:
            emission_log_probs_qgt = emission_log_probs_qgt * self.reliability.unsqueeze(1)

        emission_log_probs_gt = torch.sum(emission_log_probs_qgt, dim=0)
        return emission_log_probs_gt
    
    def get_posterior_gt(self) -> torch.Tensor:
        """ Returns the STPD from the last iteration. """
        if self.posterior_gt is None:
            raise RuntimeError("Run step_parameters() first!")
        
        posterior_gt = self.posterior_gt.clone()

        return posterior_gt
    
    def _calculate_tof_params(self) -> Dict[str, float]:
        """
        Dynamically calculates ToF boundary parameters based on environment grid and bandwidth.
        """
        import math
        LIGHT_SPEED = 299792458.0
        
        # 1. Calculate bandwidth margin (distance resolution limit)
        bw_hz = float(self.config.get('CHANNEL_BANDWIDTH_HZ', 20000000.0))
        bandwidth_margin = LIGHT_SPEED / bw_hz  
        
        # 2. Calculate room diagonal from reference grid
        min_x = torch.min(self.reference_grid[:, 0]).item()
        max_x = torch.max(self.reference_grid[:, 0]).item()
        min_y = torch.min(self.reference_grid[:, 1]).item()
        max_y = torch.max(self.reference_grid[:, 1]).item()
        room_diagonal = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        
        # 3. Calculate maximum allowed flight distance before penalty
        dist_limit = room_diagonal + bandwidth_margin
        
        # 4. Physics-derived maximum variance penalty multiplier
        # (Flattening a typical 10-degree LoS signal into a 70-degree NLoS scatter)
        nlos_spread_deg = float(self.config.get('NLOS_SPREAD_DEG', 70.0))
        los_spread_deg = float(self.config.get('LOS_SPREAD_DEG', 10.0))
        penalty_strength = (nlos_spread_deg / los_spread_deg) ** 2

        # 5. Return parameters dictionary
        return {
            'tof_limit_sec': dist_limit / LIGHT_SPEED, 
            'tof_penalty_strength': penalty_strength
        }
    
    def build_initial_state_only(self):
        """
        Build initial EPD/STPD without running Soft-EM updates.
        """
        if self.propagation_params is None:
            self.initialize_params()

        self.emission_log_probs_qgt = soft_em_utils.calculate_emission_log_probs(
            self.config, 
            self.features,
            self.propagation_params,
            self.grid_angle_qg,
            self.log_pi_qtc
        )

        emission_log_probs_gt = torch.sum(self.emission_log_probs_qgt, dim=0)

        self.posterior_gt = soft_em_utils.run_forward_backward(
            emission_log_probs_gt,
            self.neighbor_matrix,
            self.device
        )

    def _recompute_current_state(self) -> None:
        """
        Recompute emission and posterior using the current propagation parameters.
        This must be called after the final accepted parameter update so that
        the stored emission/posterior are consistent with self.propagation_params.
        """
        if self.propagation_params is None:
            raise RuntimeError("Propagation parameters are not initialized.")

        self.emission_log_probs_qgt = soft_em_utils.calculate_emission_log_probs(
            self.config,
            self.features,
            self.propagation_params,
            self.grid_angle_qg,
            self.log_pi_qtc
        )

        emission_log_probs_gt = torch.sum(self.emission_log_probs_qgt, dim=0)

        self.posterior_gt = soft_em_utils.run_forward_backward(
            emission_log_probs_gt,
            self.neighbor_matrix,
            self.device,
        )