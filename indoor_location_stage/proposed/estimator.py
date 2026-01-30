# indoor_location_stage/proposed/estimator.py

import torch
import numpy as np
from typing import Dict, Any, Optional

from scipy.signal import savgol_filter

# Import Physics Engine (SoftEM) and Utils
from .soft_em import SoftEM_Algorithm
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
        emission_gating: Optional[torch.Tensor] = None,
        transition_gating: Optional[torch.Tensor] = None
    ):
        self.features = features
        self.config = config
        self.reference_grid = reference_grid
        self.ap_data = ap_data
        self.device = device
        self.emission_gating = emission_gating
        self.transition_gating = transition_gating
        
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

        self.neighbor_matrix = ap_data['neighbor_matrix'].to(device)
        self.grid_neighbor_delay_diff_qgk = ap_data['grid_neighbor_delay_diff_qgk'].to(device)

        # State tracking
        self.trajectory = None
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
        self.softem.set_emission_gating(self.emission_gating)
        self.softem.step_parameters()
        
        # --- Step 2: Retrieve the Calculated EPD & STPD ---
        self.epd = self.softem.get_final_epd()
        self.stpd = self.softem.get_final_stpd()

        self.tpd = soft_em_utils.calculate_transition_log_probs_tof(
            features=self.features,
            neighbor_index_matrix=self.neighbor_matrix,
            grid_neighbor_delay_diff_qgk=self.grid_neighbor_delay_diff_qgk,
            transition_gating=self.transition_gating, 
            device=self.device,
        )

        # --- Step 3: AI-Assisted Trajectory Estimation (Viterbi) ---
        # Run Viterbi once with the AI Transition Handler
        raw_trajectory, _ = self.viterbi.run(
            emission_log_probs=self.epd,
            neighbor_index_matrix=self.neighbor_matrix,
            get_max_previous_score=soft_em_utils.get_max_previous_score,
            transition_log_probs=self.tpd
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
        np.save(os.path.join(debug_dir, "tpd.npy"), self.tpd.detach().cpu().numpy())

        # 2. save Grid
        np.save(os.path.join(debug_dir, "grid.npy"), self.reference_grid.detach().cpu().numpy())
        
        # 3. save SoftEM params
        params_numpy = {k: v.detach().cpu().numpy() for k, v in self.softem.propagation_params.items()}
        np.save(os.path.join(debug_dir, "softem_params.npy"), params_numpy)

        # 4. save gating
        if self.emission_gating is not None:
            np.save(os.path.join(debug_dir, "emission_gating.npy"), self.emission_gating.detach().cpu().numpy())
        if self.transition_gating is not None:
            np.save(os.path.join(debug_dir, "transition_gating.npy"), self.transition_gating.detach().cpu().numpy())

        print(f"[Estimator] Debug data saved to {debug_dir}")
        # =========================================================
        if hasattr(self.softem, "_debug_epd") and "mixed_log_prob_qgt" in self.softem._debug_epd:
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