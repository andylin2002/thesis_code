# engines/symbolic_engine/stages/result_estimation/baseline/estimator.py

import torch
import numpy as np
from typing import Dict, Any, Optional

from .hard_em import HardEMAlgorithm
from . import hard_em_utils
from .._common.viterbi import Viterbi_Algorithm
from .._common import grid_tools

class BaselineEstimator:
    """
    Controller for the Baseline method.
    Coordinates the HardEM parameter optimizer and the Viterbi path finder.
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
        self.device = device
        self.ap_data = ap_data
        
        self.num_sample = config['NUM_SAMPLE']
        self.G = reference_grid.shape[0]

        # 1. Initialize Propagation Parameter Optimizer
        self.param_optimizer = HardEMAlgorithm(
            features, config, reference_grid, ap_data, device
        )

        # 2. Initialize Trajectory Finder (Viterbi Engine)
        self.viterbi = Viterbi_Algorithm(
            self.G, self.num_sample, reference_grid, device
        )

        # 3. Pre-calculate Neighbor Matrix (Static geometric info)
        G_index = torch.arange(self.G).to(device)
        self.neighbor_matrix = grid_tools.get_all_neighbor_indices(config, G_index, device)

        # State tracking
        self.trajectory = None
        self.MEPLL_record = -torch.inf

    def solve(self) -> torch.Tensor:
        """
        Execute the main EM loop: iteratively update parameters and trajectory.
        Returns:
            Final estimated trajectory (T, 2).
        """
        # --- Initialization ---
        # 1. Initialize propagation parameters via Optimizer
        self.param_optimizer.initialize_params()
        
        # 2. Initialize trajectory (Warm-up)
        current_params = self.param_optimizer.propagation_params
        
        emission_probs = hard_em_utils.calculate_emission_probability(
            self.features,
            self.reference_grid,
            current_params,
            self.ap_data['locations'],
            self.ap_data['orientations'],
            self.device
        )
        
        self.trajectory, _ = self.viterbi.run(
            emission_log_probs=emission_probs,
            neighbor_index_matrix=self.neighbor_matrix,
            get_max_previous_score=hard_em_utils.get_max_previous_score
        )
        
        # --- Main EM Loop ---
        for i in range(self.config['EM_MAX_ITER']):
            
            # --- Step A: Parameter Update (E-Step equivalent) ---
            # Optimize parameters based on the current trajectory estimate
            mepll_params = self.param_optimizer.step_parameters(self.trajectory)
            
            # --- Step B: Trajectory Update (M-Step equivalent) ---
            # 1. Retrieve updated parameters
            current_params = self.param_optimizer.propagation_params
            
            # 2. Calculate Emission Probabilities
            emission_probs = hard_em_utils.calculate_emission_probability(
                self.features,
                self.reference_grid,
                current_params,
                self.ap_data['locations'],
                self.ap_data['orientations'],
                self.device
            )
            
            # 3. Run Viterbi to find the best path
            self.trajectory, mepll_traj = self.viterbi.run(
                emission_log_probs=emission_probs,
                neighbor_index_matrix=self.neighbor_matrix,
                get_max_previous_score=hard_em_utils.get_max_previous_score,
            )
            
            # --- Step C: Convergence Check ---
            total_mepll = mepll_params + mepll_traj
            
            if self._check_convergence(total_mepll):
                print(f"[Estimator] Converged at iter {i}")
                break
                
        return self.trajectory

    def _check_convergence(self, current_mepll: float) -> bool:
        diff = abs(self.MEPLL_record - current_mepll)
        self.MEPLL_record = current_mepll
        return diff < 1e-6