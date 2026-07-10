# engines/symbolic_engine/strategies/estimation_strategy.py

import torch
import os
import numpy as np
from typing import Optional, Tuple, Union
from core.interfaces import IEstimator
from ..stages.result_estimation._common import grid_tools
from ..stages.result_estimation.proposed import soft_em_utils

from ..stages.gating_evaluation.evaluator import GatingEvaluator
from ..stages.result_estimation.baseline.estimator import BaselineEstimator
from ..stages.result_estimation.proposed.estimator import ProposedEstimator

class BaselineEstimatorStrategy(IEstimator):
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
        self.ap_data = {
            'locations': grid_tools.get_ap_locations(config, self.num_ap, device),
            'orientations': torch.tensor(
                [data.get('ORIENTATION_DEG', 0) for data in config['ACCESS_POINTS'].values()], 
                dtype=torch.float32, 
                device=device
            )
        }

    def estimate(
        self, 
        features: torch.Tensor, 
        aggregated_csi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        estimator = BaselineEstimator(
            features=features,
            config=self.config,
            reference_grid=self.reference_grid,
            ap_data=self.ap_data, 
            device=self.device
        )
        
        trajectory = estimator.solve()
            
        return trajectory

class ProposedEstimatorStrategy(IEstimator):
    """
    Strategy for Proposed localization (Physics-Aware + optional MLP gating)
    """
    def __init__(self, config, device, reference_grid, directions_vectors):
        self.config = config
        self.device = device
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors

        self.gating_evaluator = GatingEvaluator(config, device)

        # Pre-calculate AP info ONCE during initialization
        self.num_ap = len(config['ACCESS_POINTS'])

        ap_locations = grid_tools.get_ap_locations(config, self.num_ap, device)
        ap_orientations = torch.tensor(
            [data.get('ORIENTATION_DEG', 0) for data in config['ACCESS_POINTS'].values()], 
            dtype=torch.float32, 
            device=device
        )

        grid_angle_qg = soft_em_utils.calculate_grid_angle_qg(
            reference_grid, ap_locations, ap_orientations
        ).to(device)

        grid_delay_qg = soft_em_utils.calculate_grid_delay_qg(
            reference_grid, ap_locations
        ).to(device)

        G_index = torch.arange(reference_grid.shape[0], device=device)
        neighbor_matrix = grid_tools.get_all_neighbor_indices(config, G_index, device)
        #DEBUG
        # Save neighbor_matrix once
        os.makedirs("output", exist_ok=True)
        neighbor_path = os.path.join("output", "neighbor_matrix.npy")
        if not os.path.exists(neighbor_path):
            np.save(neighbor_path, neighbor_matrix.detach().cpu().numpy())
            print(f"[ProposedStrategy] neighbor_matrix saved to {neighbor_path}")

        grid_neighbor_delay_diff_qgk = soft_em_utils.calculate_grid_neighbor_delay_diff_qgk(
            grid_delay_qg=grid_delay_qg,
            neighbor_index_matrix=neighbor_matrix,
        ).to(device)

        self.ap_data = {
            'locations': ap_locations,
            'orientations': ap_orientations,
            'grid_angle_qg': grid_angle_qg,
            'grid_delay_qg': grid_delay_qg,
            'neighbor_matrix': neighbor_matrix, 
            'grid_neighbor_delay_diff_qgk': grid_neighbor_delay_diff_qgk,
        }

        # Neural's input
        self.emission_log_probs_qgt = None
        self.posterior_gt = None
        self.reliability = None
    
    # =========================================================
    # Public API for Symbolic Worker
    # =========================================================
    def set_gating_state_dict(self, state_dict: Optional[dict], step: Optional[int] = None) -> None:
        if state_dict is None:
            print("[ProposedStrategy] Received empty gating state_dict.")
            return

        self.gating_evaluator.load_state_dict(state_dict)

    # =========================================================
    # Main estimation
    # =========================================================
    def estimate(
        self, 
        features: torch.Tensor, 
        aggregated_csi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pipeline: 
        1. AI Evaluation -> Get Weights
        2. Physics Estimation -> Use Weights -> Get Trajectory
        """
        # Get gating weight
        reliability: Optional[torch.Tensor] = None

        if aggregated_csi is not None and self.gating_evaluator.is_ready():
            try:
                reliability = self.gating_evaluator.evaluate(aggregated_csi)
            except Exception as e:
                reliability = None
                print(f"[ProposedStrategy] AI Gating failed: {e}. Using pure physics.")
        
        # Instantiate the Proposed Estimator
        estimator = ProposedEstimator(
            features=features,
            config=self.config,
            reference_grid=self.reference_grid,
            ap_data=self.ap_data,
            device=self.device,
            reliability=reliability
        )

        trajectory = estimator.solve()

        # Neural's input
        self.emission_log_probs_qgt = getattr(estimator, 'emission_log_probs_qgt', None)
        self.posterior_gt = getattr(estimator, 'posterior_gt', None)
        self.reliability = reliability
            
        return trajectory