# indoor_location_stage/proposed/estimator.py

import torch
import numpy as np
from typing import Dict, Any, List, Optional

# Import Physics Engine (SoftEM) and Utils
from .soft_em import SoftEM_Algorithm
from . import soft_em_utils
from .._common.viterbi import Viterbi_Algorithm
from .._common import grid_tools

from scipy.signal import savgol_filter

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
        self.stpd = None

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

        print(self.param_optimizer.propagation_params)
        
        # --- Step 2: Retrieve the Calculated EPD & STPD ---
        self.epd = self.param_optimizer.get_final_epd()
        self.stpd = self.param_optimizer.get_final_stpd()

        # --- Step 3: AI-Assisted Trajectory Estimation (Viterbi) ---
        # Run Viterbi once with the AI Transition Handler
        raw_trajectory, _ = self.viterbi.run(
            emission_log_probs=self.epd,
            neighbor_index_matrix=self.neighbor_matrix,
            get_max_previous_score=soft_em_utils.get_max_previous_score,
            
            # **kwargs
            model=self.model,
            feature_buffer=self.buffer,
            spd=self.spd,
            mode='TRANSFORMER'
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
        params_numpy = {k: v.detach().cpu().numpy() for k, v in self.param_optimizer.propagation_params.items()}
        np.save(os.path.join(debug_dir, "softem_params.npy"), params_numpy)

        print(f"[Estimator] Debug data saved to {debug_dir}")
        # =========================================================

        # DEBUG BEGIN
        import json, os
        import torch

        def _entropy(p, eps=1e-12):
            return -(p * (p + eps).log()).sum(dim=0)  # (T,)

        def _top1_top2_gap(logp):
            # logp: (G,T)
            top2 = torch.topk(logp, k=2, dim=0).values  # (2,T)
            gap = top2[0] - top2[1]                     # (T,)
            return gap

        def _argmax_xy(logp, grid):
            # logp: (G,T), grid: (G,2)
            idx = torch.argmax(logp, dim=0)            # (T,)
            xy = grid[idx]                              # (T,2)
            return idx, xy

        def _jump_rate(traj_xy, grid_spacing_guess=0.5):
            # traj_xy: (T,2)
            step = torch.norm(traj_xy[1:] - traj_xy[:-1], dim=1)  # (T-1,)
            thr = 3.0 * grid_spacing_guess
            jump = (step > thr).float()
            return {
                "jump_rate": jump.mean().item(),
                "max_step": step.max().item(),
                "p95_step": torch.quantile(step, 0.95).item()
            }

        debug = {}
        logEPD = self.epd.detach()
        STPD  = self.stpd.detach()

        grid = self.reference_grid.detach()

        # 1) EPD peak / gap
        epd_gap = _top1_top2_gap(logEPD)                         # (T,)
        epd_idx, epd_xy = _argmax_xy(logEPD, grid)               # (T,), (T,2)

        # 2) STPD entropy / peak
        stpd_entropy = _entropy(STPD)                            # (T,)
        stpd_idx, stpd_xy = _argmax_xy(STPD.log(), grid)         # 用 log 只是為了 argmax，一樣

        # 3) Trajectory jump rate（raw 與 smooth 都算）
        raw_traj = raw_trajectory.detach()
        smooth_traj = self.trajectory.detach()

        debug["epd_gap_mean"] = epd_gap.mean().item()
        debug["epd_gap_p95"]  = torch.quantile(epd_gap, 0.95).item()
        debug["stpd_entropy_mean"] = stpd_entropy.mean().item()
        debug["stpd_entropy_p95"]  = torch.quantile(stpd_entropy, 0.95).item()

        debug["epd_peak_xy_first10"]  = epd_xy[:10].cpu().tolist()
        debug["stpd_peak_xy_first10"] = stpd_xy[:10].cpu().tolist()

        debug["raw_jump"]   = _jump_rate(raw_traj, grid_spacing_guess=0.5)    # 0.5 先猜，之後你可換成 config 的 grid spacing
        debug["smooth_jump"] = _jump_rate(smooth_traj, grid_spacing_guess=0.5)

        # 4) SoftEM params 也一起寫（看是否爆掉）
        p = self.param_optimizer.propagation_params
        debug["params"] = {k: v.detach().cpu().tolist() for k, v in p.items()}

        # 5) 存成 json
        debug_path = os.path.join(debug_dir, "metrics.json")
        with open(debug_path, "w") as f:
            json.dump(debug, f, indent=2)
        print(f"[Estimator] Metrics saved to {debug_path}")

        dbg = self.param_optimizer._debug_epd
        # Always-present (AoA-only debug)
        if "mixed_log_prob_qgt" in dbg:
            np.save(os.path.join(debug_dir, "mixed_log_prob_qgt.npy"), dbg["mixed_log_prob_qgt"].detach().cpu().numpy())
        if "var_aoa_qtc" in dbg:
            np.save(os.path.join(debug_dir, "var_aoa_qtc.npy"), dbg["var_aoa_qtc"].detach().cpu().numpy())
        if "penalty_qtc" in dbg:
            np.save(os.path.join(debug_dir, "penalty_qtc.npy"), dbg["penalty_qtc"].detach().cpu().numpy())
        if "gain_qtc" in dbg:
            np.save(os.path.join(debug_dir, "gain_qtc.npy"), dbg["gain_qtc"].detach().cpu().numpy())

        # Optional (only exists if you still compute ToF-related debug somewhere)
        if "var_tof_qtc" in dbg:
            np.save(os.path.join(debug_dir, "var_tof_qtc.npy"), dbg["var_tof_qtc"].detach().cpu().numpy())
        if "var_tof_base_qtc" in dbg:
            np.save(os.path.join(debug_dir, "var_tof_base_qtc.npy"), dbg["var_tof_base_qtc"].detach().cpu().numpy())
        if "tof_inflation_qtc" in dbg:
            np.save(os.path.join(debug_dir, "tof_inflation_qtc.npy"), dbg["tof_inflation_qtc"].detach().cpu().numpy())


        mixed = dbg["mixed_log_prob_qgt"]  # (Q,G,T)
        g_star = torch.argmax(self.epd, dim=0)          # (T,)
        # 取每個 t，在 g* 的每個 AP 貢獻
        contrib_qt = mixed[:, g_star, torch.arange(mixed.shape[2], device=mixed.device)]  # (Q,T)
        top_ap = torch.argmax(contrib_qt, dim=0)  # (T,)

        # 存前 10 步看看是不是某個 AP 一直主導
        debug["top_ap_first10"] = top_ap[:10].detach().cpu().tolist()
        debug["top_ap_contrib_first10"] = contrib_qt[:, :10].detach().cpu().tolist()

        LIGHT_SPEED = 299792458.0

        p = os.path.join(debug_dir, "var_tof_qtc.npy")
        if os.path.exists(p):
            var_tof = np.load("output/debug/var_tof_qtc.npy")  # (Q,T,C) sec^2
            var_tof_t = torch.from_numpy(var_tof)

            std_tof_m = torch.sqrt(var_tof_t) * LIGHT_SPEED
            print("[ToF STD] median(m):", std_tof_m.median().item())
            print("[ToF STD] p95(m):", torch.quantile(std_tof_m.flatten(), 0.95).item())
        else:
            print("[ToF] var_tof_qtc.npy not found (ToF not used in EPD).")

        features = np.load("output/csi_features.npy")  # (Q,T,C,5)
        features_t = torch.from_numpy(features)

        tof_sprd_sec = features_t[..., 3]
        tof_sprd_m = tof_sprd_sec * LIGHT_SPEED
        print("[ToF SPRD] median(m):", tof_sprd_m.median().item())
        print("[ToF SPRD] p95(m):", torch.quantile(tof_sprd_m.flatten(), 0.95).item())



        tof_sprd = features[..., 3]  # (Q,T,C)
        # ---- unify to numpy ----
        if isinstance(tof_sprd, torch.Tensor):
            tof_sprd_m = (tof_sprd * LIGHT_SPEED).detach().cpu().numpy()
        else:
            tof_sprd_m = tof_sprd * LIGHT_SPEED  # already numpy

        flat = tof_sprd_m.reshape(-1)

        print("[ToF SPRD(m)] median:", float(np.median(flat)))
        print("[ToF SPRD(m)] p95:", float(np.quantile(flat, 0.95)))

        for th in [1.0, 2.0, 3.0, 5.0]:
            ratio = float(np.mean(flat < th))
            print(f"[ToF good ratio] sprd<{th}m:", ratio)

        # per time: best candidate per AP -> then best AP
        tof_sprd_m_qt = np.min(tof_sprd_m, axis=2)   # (Q,T)
        tof_sprd_m_t  = np.min(tof_sprd_m_qt, axis=0) # (T,)

        th = 2.0
        good = (tof_sprd_m_t < th).astype(np.int32)

        runs = []
        cur = 0
        for v in good:
            if v == 1:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)

        if len(runs) == 0:
            print(f"[Good runs sprd<{th}m] none")
        else:
            print(f"[Good runs sprd<{th}m] count:", len(runs))
            print(f"[Good runs sprd<{th}m] mean_len:", float(np.mean(runs)))
            print(f"[Good runs sprd<{th}m] max_len:", int(np.max(runs)))


        # DEBUG END

        return self.trajectory
    
    # DEBUG
    def _apply_physics_smoothing(self, raw_coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Savitzky-Golay filter to smooth the trajectory, enforcing inertia.
        Input: raw_coords (Tensor [T, 2]) on GPU
        Output: smooth_coords (Tensor [T, 2]) on GPU
        """
        # 1. Convert to CPU Numpy
        coords_np = raw_coords.detach().cpu().numpy()
        T = coords_np.shape[0]

        # 2. Parameters (根據您的採樣率調整)
        # 假設 T=100 (1秒)，window設 11~31 是合理的
        # window_length 必須是奇數，且小於 T
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