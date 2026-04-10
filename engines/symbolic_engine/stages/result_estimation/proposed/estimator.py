# engines/symbolic_engine/stages/result_estimation/proposed/estimator.py

import torch
from typing import Dict, Any, Optional

# Import Physics Engine (SoftEM) and Utils
from .soft_em import SoftEMAlgorithm
from . import soft_em_utils
from .._common.viterbi import Viterbi_Algorithm

class ProposedEstimator:
    """
    Controller for the Proposed Method (Physics-Aware AI).
    """
    def __init__(
        self, 
        features: torch.Tensor, 
        config: Dict[str, Any], 
        reference_grid: torch.Tensor,
        ap_data: Dict[str, Any],      # Pre-calculated AP info
        device: torch.device,
        reliability: Optional[torch.Tensor] = None,
    ):
        self.features = features
        self.config = config
        self.reference_grid = reference_grid
        self.ap_data = ap_data
        self.device = device
        self.reliability = reliability
        
        self.num_sample = config['NUM_SAMPLE']
        self.G = reference_grid.shape[0]

        # 1. Initialize Physics Optimizer (Soft EM)
        self.softem = SoftEMAlgorithm(
            features, config, reference_grid, ap_data, device
        )

        # 2. Initialize Trajectory Finder (Viterbi Engine)
        self.viterbi = Viterbi_Algorithm(
            self.G, self.num_sample, reference_grid, device
        )

        self.neighbor_matrix = ap_data['neighbor_matrix'].to(device)
        self.grid_neighbor_delay_diff_qgk = ap_data['grid_neighbor_delay_diff_qgk'].to(device)

        # State tracking
        self.trajectory = None
        self.epd_qgt = None
        self.epd = None
        self.stpd = None
        self.tpd = None

    def solve(self) -> torch.Tensor:
        """
        Execute the linear Physics-Aware pipeline.
        Returns:
            Final estimated trajectory (T, 2).
        """
        # --- Initialization ---
        self.softem.initialize_params()
        
        # --- Step 1: Physics Parameter Optimization (SoftEM) ---
        # This function runs the internal EM loop until parameters converge.
        # Process: Init -> [Calc EPD -> Calc Gamma -> Update Params] * N -> Converged Parameters
        self.softem.set_reliability(self.reliability)
        
        use_soft_em = self.config.get("USE_SOFT_EM", True)
        if use_soft_em:
            self.softem.step_parameters()
        else:
            self.softem.build_initial_state_only()
        
        # --- Step 2: Retrieve the Calculated EPD & STPD ---
        self.epd_qgt = self.softem.get_epd_qgt(apply_reliability=False) #DEBUG
        self.epd = self.softem.get_final_epd()
        self.stpd = self.softem.get_final_stpd()
        self.tpd = None

        # --- Step 3: AI-Assisted Trajectory Estimation (Viterbi) ---
        # Run Viterbi once with the AI Transition Handler
        trajectory, _ = self.viterbi.run(
            emission_log_probs=self.epd,
            neighbor_index_matrix=self.neighbor_matrix,
            get_max_previous_score=soft_em_utils.get_max_previous_score,
            transition_log_probs=self.tpd
        )

        self.trajectory = trajectory

        return self.trajectory