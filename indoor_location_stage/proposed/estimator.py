# indoor_location_stage/proposed/estimator.py

import torch
import numpy as np
from typing import Dict, Any, List, Optional

from scipy.signal import savgol_filter

# Import Physics Engine (SoftEM) and Utils
from .soft_em import SoftEM_Algorithm
from . import soft_em_utils
from .._common.viterbi import Viterbi_Algorithm
from .._common import grid_tools

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
        ap_time_gate: Optional[torch.Tensor] = None
    ):
        self.features = features
        self.config = config
        self.reference_grid = reference_grid
        self.ap_data = ap_data
        self.device = device
        self.ap_time_gate = ap_time_gate
        
        self.num_sample = config['NUM_SAMPLE']
        self.G = reference_grid.shape[0]

        # 1. Initialize Physics Optimizer (Soft EM)
        self.softem = SoftEM_Algorithm(
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
        self.stpd = None

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
        self.softem.set_ap_time_gate(self.ap_time_gate)
        self.softem.step_parameters()
        
        # --- Step 2: Retrieve the Calculated EPD & STPD ---
        self.epd = self.softem.get_final_epd()
        self.stpd = self.softem.get_final_stpd()

        # --- Step 3: AI-Assisted Trajectory Estimation (Viterbi) ---
        # Run Viterbi once with the AI Transition Handler
        raw_trajectory, _ = self.viterbi.run(
            emission_log_probs=self.epd,
            neighbor_index_matrix=self.neighbor_matrix,
            get_max_previous_score=soft_em_utils.get_max_previous_score,
        )

        self.trajectory = self._apply_physics_smoothing(raw_trajectory)

        # =========================================================
        # [SAVE FOR ANALYSIS]
        # =========================================================
        import os
        import numpy as np
        
        debug_dir = "output/debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # 1. save EPD & STPD
        np.save(os.path.join(debug_dir, "epd.npy"), self.epd.detach().cpu().numpy())
        np.save(os.path.join(debug_dir, "stpd.npy"), self.stpd.detach().cpu().numpy())
        
        # 2. save Grid
        np.save(os.path.join(debug_dir, "grid.npy"), self.reference_grid.detach().cpu().numpy())
        
        # 3. save SoftEM params
        params_numpy = {k: v.detach().cpu().numpy() for k, v in self.softem.propagation_params.items()}
        np.save(os.path.join(debug_dir, "softem_params.npy"), params_numpy)

        # 4. save gating
        if self.ap_time_gate is not None:
            np.save(os.path.join(debug_dir, "ap_time_gate.npy"), self.ap_time_gate.detach().cpu().numpy())


        print(f"[Estimator] Debug data saved to {debug_dir}")
        # =========================================================

        dbg = self.softem._debug_epd
        np.save(os.path.join(debug_dir, "mixed_log_prob_qgt.npy"), dbg["mixed_log_prob_qgt"].detach().cpu().numpy())

        return self.trajectory
    
    def _apply_physics_smoothing(self, raw_coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Savitzky-Golay filter to smooth the trajectory, enforcing inertia.
        Input: raw_coords (Tensor [T, 2]) on GPU
        Output: smooth_coords (Tensor [T, 2]) on GPU
        """
        # 1. Convert to CPU Numpy
        coords_np = raw_coords.detach().cpu().numpy()
        T = coords_np.shape[0]

        # 2. Parameters Setup
        window_length = 11
        polyorder = 2

        if T <= window_length:
            return raw_coords

        try:
            # 3. Apply Filter
            smooth_np = savgol_filter(coords_np, window_length, polyorder, axis=0)
            
            # 4. Convert back to Tensor
            smooth_coords = torch.from_numpy(smooth_np).to(raw_coords.device).type(raw_coords.dtype)
            return smooth_coords

        except Exception as e:
            print(f"[Warning] Smoothing failed: {e}. Using raw trajectory.")
            return raw_coords