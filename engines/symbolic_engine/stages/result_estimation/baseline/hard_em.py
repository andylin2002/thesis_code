# engines/symbolic_engine/stages/result_estimation/baseline/hard_em.py

import torch
import numpy as np
from typing import Dict, Any, Optional

from . import hard_em_utils
from .._common import math_tools
from .._common import grid_tools

# Type definition for clarity
TypePropParams = Dict[str, torch.Tensor]

class HardEMAlgorithm:
    """
    Parameter Optimizer for the Baseline Method.
    Refactored to focus solely on estimating propagation parameters (E-Step)
    given a fixed trajectory.
    """
    def __init__(
            self, 
            features: torch.Tensor, 
            config: Dict[str, Any], 
            reference_grid: torch.Tensor,
            ap_data: Dict[str, Any], # Received from Estimator
            device: torch.device
        ):
        
        self.features = features
        self.config = config
        self.reference_grid = reference_grid
        self.device = device
        
        # Unpack AP information (Passed from Estimator to avoid re-calculation)
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.ap_locations = ap_data['locations']
        self.ap_orientations = ap_data['orientations']

        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']
        
        # State: Propagation Parameters and Score
        self.propagation_params: Optional[TypePropParams] = None
        self.MEPLL_PropParams = -torch.inf

    def initialize_params(self) -> TypePropParams:
        """
        Initialize propagation parameters (Alpha, Beta, etc.) based on features.
        Matches original _initialize_PropParams logic.
        """
        Q = self.num_ap
        T = self.num_sample
        K = 2 # LOS and NLOS
        MIN_VAR = 1

        # Initialize tensors
        alpha_qk =      torch.zeros(Q, K, dtype=torch.float32, device=self.device)
        beta_qk =       torch.zeros(Q, K, dtype=torch.float32, device=self.device)
        power_qk_var =  torch.zeros(Q, K, dtype=torch.float32, device=self.device)
        angle_k_var =   torch.zeros(K, dtype=torch.float32, device=self.device)
        delay_k_mean =  torch.zeros(K, dtype=torch.float32, device=self.device)
        delay_k_var =   torch.zeros(K, dtype=torch.float32, device=self.device)
        pi_k =          torch.full((K, ), 0.5, dtype=torch.float32, device=self.device)
        gamma_qtk =     torch.full((Q, T, K), 0.5, dtype=torch.float32, device=self.device)

        # Extract features
        power_qt = self.features[:, :, 0]
        angle_qt = self.features[:, :, 1]
        delay_qt = self.features[:, :, 2]

        # --- Initialize POWER parameters ---
        alpha_qk, beta_qk = hard_em_utils.calculate_init_alpha_and_beta_qk(
            self.reference_grid, self.ap_locations, power_qt
        )
        power_qk_var = hard_em_utils.calculate_init_power_qk_var(
            self.reference_grid, self.ap_locations, alpha_qk, beta_qk, power_qt
        )

        # --- Initialize ANGLE parameters ---
        angle_k_var = hard_em_utils.calculate_init_angle_k_var(
            self.reference_grid, self.ap_locations, self.ap_orientations, angle_qt
        )

        # --- Initialize DELAY parameters ---
        delay_flat = delay_qt.flatten()
        delay_means, delay_vars = hard_em_utils.estimate_two_gaussians(delay_flat)
        delay_k_mean.copy_(delay_means.to(self.device))
        delay_k_var.copy_(delay_vars.clamp(min=MIN_VAR))

        # --- Initialize Gamma ---
        delay_11k_mean = delay_k_mean.view(1, 1, -1)
        delay_11k_var = delay_k_var.view(1, 1, -1)
        mean_distance = torch.abs(delay_k_mean[0] - delay_k_mean[1])
        delay_11k_var = delay_11k_var * 1 * mean_distance.clamp(min=1.0)
        
        delay_distribution_qtk = hard_em_utils.build_gaussian_distribution(
            delay_11k_mean, delay_11k_var
        )
        gamma_qtk = hard_em_utils.calculate_init_gamma_qtk(
            delay_distribution_qtk, delay_qt
        )
        
        # Structure results
        propagation_params = {
            'alpha_qk':             alpha_qk,
            'beta_qk':              beta_qk,
            'power_qk_var':         power_qk_var,
            'angle_k_var':          angle_k_var,
            'delay_k_mean':         delay_k_mean,
            'delay_k_var':          delay_k_var,
            'pi_k':                 pi_k,
            'gamma_qtk':            gamma_qtk,
        }
        
        # Save internally
        self.propagation_params = propagation_params
        return self.propagation_params

    def step_parameters(self, trajectory: torch.Tensor) -> float:
        """
        Perform the E-step (Parameter Update).
        Optimizes propagation parameters to maximize likelihood given the input trajectory.
        
        Args:
            trajectory: Current estimated path coordinates (T, 2).
        Returns:
            MEPLL score for the optimized parameters.
        """
        # Ensure we have initial params
        if self.propagation_params is None:
            self.initialize_params()

        propagation_params = self.propagation_params
        
        power_qt = self.features[:, :, 0]
        angle_qt = self.features[:, :, 1]
        delay_qt = self.features[:, :, 2]
        
        # --- Pre-calculation based on Trajectory ---
        L_qt = hard_em_utils.calculate_L_qt(trajectory, self.ap_locations)
        angle_qt_mean = hard_em_utils.calculate_angle_qt_mean(
            trajectory, self.ap_locations, self.ap_orientations
        )
        angle_qt1_mean = angle_qt_mean.unsqueeze(2)

        # --- Inner Loop: Maximize MEPLL until convergence ---
        MAX_MEPLL_PropParams = -torch.inf

        # Optimization loop for parameters (Inner EM Logic)
        while True:
            # Snapshot for rollback if needed
            propagation_params_old = {k: v.clone() for k, v in propagation_params.items()}

            # Weighted averages
            power_qk_average = math_tools.calculate_weighted_average(
                data=power_qt.unsqueeze(2), weights=propagation_params['gamma_qtk'], dim=1
            )
            L_qk_average = math_tools.calculate_weighted_average(
                data=L_qt.unsqueeze(2), weights=propagation_params['gamma_qtk'], dim=1
            )

            # --- Update Power Params ---
            propagation_params['alpha_qk'] = hard_em_utils.calculate_alpha_qk(
                power_qt, power_qk_average, L_qt, L_qk_average, propagation_params['gamma_qtk']
            )
            propagation_params['beta_qk'] = hard_em_utils.calculate_beta_qk(
                propagation_params['alpha_qk'], power_qk_average, L_qk_average
            )
            power_qtk_mean = hard_em_utils.calculate_power_qtk_mean(
                propagation_params['alpha_qk'], propagation_params['beta_qk'], L_qt
            )
            propagation_params['power_qk_var'] = hard_em_utils.calculate_power_qk_var(
                power_qt, power_qtk_mean, propagation_params['gamma_qtk']
            ) 
            
            power_distribution_qtk = hard_em_utils.build_gaussian_distribution(
                power_qtk_mean, propagation_params['power_qk_var'].unsqueeze(1)
            )
            
            # --- Update Angle Params ---
            propagation_params['angle_k_var'] = hard_em_utils.calculate_angle_k_var(
                angle_qt_mean, angle_qt, propagation_params['gamma_qtk']
            )
            angle_distribution_qtk = hard_em_utils.build_gaussian_distribution(
                angle_qt1_mean, propagation_params['angle_k_var'].view(1, 1, -1)
            )

            # --- Update Delay Params ---
            propagation_params['delay_k_mean'] = hard_em_utils.calculate_delay_k_mean(
                delay_qt, propagation_params['gamma_qtk']
            )
            propagation_params['delay_k_var'] = hard_em_utils.calculate_delay_k_var(
                propagation_params['delay_k_mean'], delay_qt, propagation_params['gamma_qtk']
            )
            delay_distribution_qtk = hard_em_utils.build_gaussian_distribution(
                propagation_params['delay_k_mean'].view(1, 1, -1), 
                propagation_params['delay_k_var'].view(1, 1, -1)
            )
            
            # --- Update Global LOS ratio ---
            propagation_params['pi_k'] = hard_em_utils.calculate_pi(propagation_params['gamma_qtk'])

            # --- Calculate MEPLL ---
            MEPLL_PropParams_new = hard_em_utils.calculate_MEPLL_PropParams(
                propagation_params['pi_k'], 
                power_distribution_qtk, 
                angle_distribution_qtk, 
                delay_distribution_qtk, 
                power_qt, angle_qt, delay_qt,
            )

            # --- Update AP LOS ratio (Gamma) ---
            propagation_params['gamma_qtk'] = hard_em_utils.calculate_gamma_qtk(
                propagation_params['pi_k'], 
                power_distribution_qtk, 
                angle_distribution_qtk, 
                delay_distribution_qtk, 
                power_qt, angle_qt, delay_qt
            )

            # --- Convergence Check (Inner Loop) ---
            if MEPLL_PropParams_new > MAX_MEPLL_PropParams + 1e-4:
                MAX_MEPLL_PropParams = MEPLL_PropParams_new
                continue
            else:
                # Revert to old params if not improving
                self.propagation_params = propagation_params_old 
                break
        
        # Final update of state
        self.propagation_params = propagation_params
        self.MEPLL_PropParams = MAX_MEPLL_PropParams
        
        return self.MEPLL_PropParams