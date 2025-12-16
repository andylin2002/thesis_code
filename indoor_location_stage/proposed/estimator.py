import torch
import numpy as np
from typing import Dict, Any, List, Optional

# Import Physics Engine (SoftEM) and Utils
from .soft_em import SoftEM_Algorithm
from . import soft_em_utils
from .._common.viterbi import Viterbi_Algorithm
from .._common import grid_tools

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
        self.epd = None

    def solve(self) -> torch.Tensor:
        """
        Execute the linear Physics-Aware pipeline.
        Returns:
            Final estimated trajectory (T, 2).
        """
        # --- Initialization ---
        self.param_optimizer.initialize_params()
        
        # --- Step 1: Physics Parameter Optimization (SoftEM) ---
        # This function runs the internal EM loop until parameters converge.
        # Process: Init -> [Calc EPD -> Calc Gamma -> Update Params] * N -> Converged Parameters
        self.param_optimizer.step_parameters()
        
        # --- Step 2: Retrieve the Calculated EPD ---
        self.epd = self.param_optimizer.get_final_epd()

        # --- Step 3: AI-Assisted Trajectory Estimation (Viterbi) ---
        # Run Viterbi once with the AI Transition Handler
        self.trajectory, _ = self.viterbi.run(
            emission_log_probs=self.epd,
            neighbor_index_matrix=self.neighbor_matrix,
            get_max_previous_score=soft_em_utils.get_max_previous_score,
            
            # **kwargs
            model=self.model,
            feature_buffer=self.buffer,
            spd=self.spd,
            mode='TRANSFORMER'
        )

        # =========================================================
        # 💾 [SAVE FOR DEBUG]
        # =========================================================
        import os
        import numpy as np
        
        debug_dir = "output/debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # 1. save EPD
        np.save(os.path.join(debug_dir, "epd.npy"), self.epd.detach().cpu().numpy())
        
        # 2. save Grid
        np.save(os.path.join(debug_dir, "grid.npy"), self.reference_grid.detach().cpu().numpy())
        
        # 3. save SoftEM params
        params_numpy = {k: v.detach().cpu().numpy() for k, v in self.param_optimizer.propagation_params.items()}
        np.save(os.path.join(debug_dir, "softem_params.npy"), params_numpy)

        print(f"[Estimator] Debug data saved to {debug_dir}")
        # =========================================================
        
        return self.trajectory