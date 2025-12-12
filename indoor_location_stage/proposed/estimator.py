import torch
import numpy as np
from typing import Dict, Any, List, Optional

# Import Physics Engine (SoftEM) and Utils
from .soft_em import SoftEM_Algorithm
from . import soft_em_utils
from .._common.viterbi import Viterbi_Algorithm
from .._common import grid_tools
from .._common import math_tools

class ProposedEstimator:
    """
    Controller for the Proposed Method (Physics-Aware AI).
    
    Pipeline:
    1. Physics Layer (SoftEM): Internally iterates to converge on optimal parameters.
    2. Emission Layer: Calculates EPD (Emission Probability Distribution).
    3. Fusion Layer (AI + Viterbi): Fuses EPD with Transformer predictions to find the path.
    
    Note: Unlike Baseline, this does NOT loop back. It's a one-pass flow.
    """
    def __init__(
        self, 
        features: torch.Tensor, 
        buffer: List[torch.Tensor],   # History buffer for AI
        spd: Optional[torch.Tensor],  # Statistical Phase Difference
        config: Dict[str, Any], 
        reference_grid: torch.Tensor,
        ap_data: Dict[str, Any],      # Pre-calculated AP info
        device: torch.device,
        model: Optional[torch.nn.Module] # Transformer Model
    ):
        self.features = features
        self.buffer = buffer
        self.spd = spd
        self.config = config
        self.reference_grid = reference_grid
        self.device = device
        self.ap_data = ap_data
        self.model = model
        
        self.num_sample = config['NUM_SAMPLE']
        self.G = reference_grid.shape[0]

        # 1. Initialize Physics Optimizer (Soft EM)
        self.param_optimizer = SoftEM_Algorithm(
            features, config, reference_grid, ap_data, device
        )

        # 2. Initialize Trajectory Finder (Viterbi Engine)
        self.viterbi = Viterbi_Algorithm(
            self.G, self.num_sample, reference_grid, device
        )

        # 3. Pre-calculate Neighbor Matrix
        G_index = torch.arange(self.G).to(device)
        self.neighbor_matrix = grid_tools.get_all_neighbor_indices(config, G_index, device)

        # State tracking
        self.trajectory = None
        self.epd = None  # Store EPD for Training Worker

    def solve(self) -> torch.Tensor:
        """
        Execute the linear Physics-Aware pipeline.
        Returns:
            Final estimated trajectory (T, 2).
        """
        # --- Initialization ---
        self.param_optimizer.initialize_params()
        
        # SoftEM will optimize parameters based on features (and this init path if needed)
        initial_trajectory = torch.zeros((self.num_sample, 2), device=self.device)
        
        # --- Step 1: Physics Parameter Optimization (SoftEM) ---
        # This function runs the internal EM loop until parameters converge.
        # We do NOT loop this with Viterbi.
        _ = self.param_optimizer.step_parameters(initial_trajectory)
        
        # --- Step 2: Calculate EPD (Emission) ---
        # Retrieve the optimized parameters
        current_params = self.param_optimizer.propagation_params
        
        # Calculate Soft Emission Probabilities (EPD)
        emission_probs = math_tools.calculate_emission_probability(
            self.features,
            self.reference_grid,
            current_params,
            self.ap_data['locations'],
            self.ap_data['orientations'],
            self.device
        )
        
        # [CRITICAL] Save EPD for external access (Training Worker)
        self.epd = emission_probs

        # --- Step 3: AI-Assisted Trajectory Estimation (Viterbi) ---
        # Run Viterbi once with the AI Transition Handler
        self.trajectory, _ = self.viterbi.run(
            emission_log_probs=emission_probs,
            neighbor_index_matrix=self.neighbor_matrix,
            get_max_previous_score=soft_em_utils.get_max_previous_score,
            
            # **kwargs
            model=self.model,
            feature_buffer=self.buffer,
            spd=self.spd,
            mode='TRANSFORMER'
        )
        
        return self.trajectory