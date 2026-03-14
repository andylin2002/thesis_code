# engines/symbolic_engine/stages/result_estimation/proposed/soft_em_utils.py

import torch
import math
from typing import Dict, Any, Tuple, Optional

from .._common import math_tools

LIGHT_SPEED = 299792458.0  # m/s

# ==========================================
# 1. Initialization
# ==========================================

def initialize_parameters(
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Initialize linear model parameters (Weight=1, Bias=0).
    """
    Q = len(config['ACCESS_POINTS'])
    
    aoa_weight =    torch.ones(1, dtype=torch.float32, device=device)
    aoa_bias_q =    torch.zeros(Q, dtype=torch.float32, device=device)
    aoa_offset_q =  torch.zeros(Q, dtype=torch.float32, device=device)

    return {
        'aoa_weight':   aoa_weight,
        'aoa_bias_q':   aoa_bias_q, 
        'aoa_offset_q': aoa_offset_q, 
    }

def calculate_grid_angle_qg(
    reference_grid: torch.Tensor, 
    ap_locations: torch.Tensor, 
    ap_orientations: torch.Tensor
) -> torch.Tensor:
    """
    Calculates relative AoA between every AP and every Grid point.
    
    Args:
        reference_grid: (G, 2)
        ap_locations: (Q, 2)
        ap_orientations: (Q,) in degrees
    Returns:
        angles: (Q, G) wrapped to [-180, 180]
    """
    grid_expanded = reference_grid.unsqueeze(0)         # (1, G, 2)
    ap_locations_expanded = ap_locations.unsqueeze(1)   # (Q, 1, 2)

    vec = grid_expanded - ap_locations_expanded
    x_diff = vec[..., 0]
    y_diff = vec[..., 1]

    angle_rad_qg = torch.atan2(y_diff, x_diff)
    angle_deg_qg = torch.rad2deg(angle_rad_qg)
    angle_deg_qg = torch.fmod(angle_deg_qg + 360.0, 360.0)

    ap_orientations_expanded = ap_orientations.unsqueeze(1).expand_as(angle_deg_qg)
    relative_angle_deg = math_tools.angular_error(ap_orientations_expanded, angle_deg_qg)
    grid_angle_qg = torch.clamp(relative_angle_deg, min=-90.0, max=90.0)

    return grid_angle_qg

def calculate_grid_delay_qg(
    reference_grid: torch.Tensor, 
    ap_locations: torch.Tensor
) -> torch.Tensor:
    """
    Calculates Euclidean distance between every AP and every Grid point.
    """
    grid_dists_qg = torch.cdist(ap_locations, reference_grid, p=2)
    grid_delay_qg = grid_dists_qg / LIGHT_SPEED

    return grid_delay_qg

def calculate_grid_neighbor_delay_diff_qgk(
    grid_delay_qg: torch.Tensor, 
    neighbor_index_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Returns:
        grid_neighbor_delay_diff_qgk: (Q, G, K=9)
        For each current grid g and neighbor slot k (prev g'), compute:
            delay(q,g) - delay(q,g')
        Invalid neighbors (-1) are set to 0 by default, and should be masked later if needed.
    """
    Q, G = grid_delay_qg.shape
    G2, K = neighbor_index_matrix.shape
    assert G2 == G, f"G mismatch: grid_delay_qg has {G}, neighbor_matrix has {G2}"
    assert K == 9, f"Expected K=9, got K={K}"

    device = grid_delay_qg.device
    dtype = grid_delay_qg.dtype

    # (G,9) mask
    valid_mask_g9 = (neighbor_index_matrix != -1)

    # Safe gather index: clamp -1 -> 0 (will be masked)
    neigh_safe_g9 = neighbor_index_matrix.clamp(min=0)  # (G,9)

    # delay at current g: (Q,G,1)
    delay_qg1 = grid_delay_qg.unsqueeze(-1)

    # delay at neighbor g' gathered: grid_delay_qg[:, neigh_safe_g9] -> (Q,G,9)
    # Advanced indexing works: (Q,G) indexed by (G,9) on dim=1 => (Q,G,9)
    delay_qg9_prev = grid_delay_qg[:, neigh_safe_g9]

    # diff: current - prev
    diff_qg9 = delay_qg1 - delay_qg9_prev  # (Q,G,9)

    # Mask invalid neighbors: set diff to 0 (or any value; invalid will be ignored in logits anyway)
    diff_qg9 = torch.where(
        valid_mask_g9.unsqueeze(0),
        diff_qg9,
        torch.zeros((1,), device=device, dtype=dtype),
    )

    return diff_qg9

# ==========================================
# 2. Calculate Emission log Probability Distribution (EPD)
# ==========================================

def calculate_emission_log_probs(
    features: torch.Tensor,
    params: Dict[str, torch.Tensor],
    grid_angle_qg: torch.Tensor, 
    tof_params: Dict[str, float]
) -> torch.Tensor:
    """
    Calculates AoA-based emission log-probabilities.
    """
    Q, T, C, _ = features.shape
    G = grid_angle_qg.shape[1]

    # 1. Feature Extraction & Weighting
    obs_aoa_cand = features[..., 0]
    obs_aoa_sprd = features[..., 1]
    obs_tof_cand = features[..., 2]
    obs_tof_sprd = features[..., 3]
    obs_gain_cand = features[..., 4]

    # Calculate Gain Penalty
    gain_var_penalty = _calculate_gain_variance_penalty(obs_gain_cand)

    # Calculate Independent Reliability
    log_rel_gain = _calculate_log_gain_reliability(obs_gain_cand)
    log_rel_tof = _calculate_log_tof_reliability(obs_tof_cand, obs_tof_sprd, tof_params)

    log_path_scores = log_rel_gain + log_rel_tof
    log_reliability_weights = torch.log_softmax(log_path_scores, dim=-1)

    # 2. Gaussian Parameters (Variance & Mean)
    # Variance dynamic adjustment
    bias_aoa = params['aoa_bias_q'].view(Q, 1, 1)
    var_aoa = ((params['aoa_weight'] * (obs_aoa_sprd ** 2) + bias_aoa) * gain_var_penalty).clamp(min=1e-6).unsqueeze(1)

    # Mean vectors from grid
    mean_aoa = grid_angle_qg.view(Q, G, 1, 1) + params['aoa_offset_q'].view(Q, 1, 1, 1)

    # 3. Calculate Log Probability (Per AP)
    # PDF: (Q, G, T, C)
    log_prob_aoa = _gaussian_log_pdf_angular(obs_aoa_cand.unsqueeze(1), mean_aoa, var_aoa)
    
    # Mixture Integration (LogSumExp over Candidates)
    emission_log_probs_qgt = torch.logsumexp(log_prob_aoa + log_reliability_weights.unsqueeze(1), dim=-1)

    return emission_log_probs_qgt

def _gaussian_log_pdf_angular(x, mean, var):
    """ Angular Gaussian Log PDF handling cyclic wrapping (-180 to 180) """
    diff = x - mean
    # Map difference to [-180, 180]
    diff = torch.remainder(diff + 180.0, 360.0) - 180.0
    return -0.5 * (torch.log(2 * torch.pi * var) + diff**2 / var)

def _gaussian_log_pdf_linear(x, mean, var):
    """ Standard Gaussian Log PDF: -0.5*log(2pi*var) - (x-mu)^2/(2var) """
    return -0.5 * (torch.log(2 * torch.pi * var) + (x - mean)**2 / var)

def _calculate_gain_variance_penalty(obs_gain_cand: torch.Tensor) -> torch.Tensor:
    max_gain, _ = torch.max(obs_gain_cand, dim=2, keepdim=True)
    norm_gain = obs_gain_cand / (max_gain + 1e-9)
    return 1.0 / (norm_gain + 0.1)

def _calculate_log_gain_reliability(obs_gain_cand: torch.Tensor) -> torch.Tensor:
    """
    Computes log-reliability based on path strength.
    """
    log_gain_reliability = torch.log(obs_gain_cand + 1e-9)
    
    return log_gain_reliability

def _calculate_log_tof_reliability(
    obs_tof_cand: torch.Tensor, 
    obs_tof_sprd: torch.Tensor, 
    tof_params: Dict[str, float]
) -> torch.Tensor:
    """
    Computes log-reliability based on ToF physical boundary.
    """
    limit = tof_params['tof_limit_sec']
    strength = tof_params['tof_penalty_strength']
    
    safe_sprd = torch.clamp(obs_tof_sprd, min=1e-10)
    z_score = (obs_tof_cand - limit) / safe_sprd
    p_invalid = 0.5 * (1.0 + torch.erf(z_score / 1.41421356))
    
    log_tof_reliability = torch.log((1.0 - p_invalid) + (p_invalid / strength) + 1e-9)
    return log_tof_reliability

# ==========================================
# 3. Calculate Spatial-Temporal Probability Distribution (STPD)
# ==========================================

def run_forward_backward(
    emission_log_probs: torch.Tensor,     # (G, T), log P(z_t | x_t=g) up to a constant
    neighbor_index_matrix: torch.Tensor,  # (G, K), -1 means invalid neighbor
    device: torch.device
) -> torch.Tensor:
    """
    Forward-Backward in log-space on a neighbor graph.

    Assumptions:
    - Transition is uniform over valid neighbors.
    - Degree-normalized to avoid bias toward high-degree nodes.

    Returns:
    - gamma: (G, T), posterior P(x_t=g | z_{1:T}), each column sums to 1.
    """
    G, T = emission_log_probs.shape
    invalid_val = -1e9

    # Valid neighbor mask and per-state degree
    valid_mask = (neighbor_index_matrix != -1)                 # (G, K)
    degree = valid_mask.sum(dim=1).clamp(min=1).float()        # (G,)
    log_uniform = -torch.log(degree)                           # (G,), log(1/degree)

    # Optional: uniform initial distribution
    log_pi = -torch.log(torch.tensor(float(G), device=device))

    # ---------- Forward (alpha) ----------
    log_alpha = torch.empty((G, T), device=device)
    log_alpha[:, 0] = log_pi + emission_log_probs[:, 0]

    # Safe gather: clamp -1 to 0, then mask it out
    neigh_safe = neighbor_index_matrix.clamp(min=0)

    for t in range(1, T):
        prev = log_alpha[:, t - 1]                             # (G,)
        neigh_prev = prev[neigh_safe]                          # (G, K)
        neigh_prev = torch.where(
            valid_mask, neigh_prev, torch.full_like(neigh_prev, invalid_val)
        )

        log_trans = torch.logsumexp(neigh_prev, dim=1) + log_uniform
        log_alpha[:, t] = emission_log_probs[:, t] + log_trans

    # ---------- Backward (beta) ----------
    log_beta = torch.zeros((G, T), device=device)

    for t in range(T - 2, -1, -1):
        next_pot = log_beta[:, t + 1] + emission_log_probs[:, t + 1]  # (G,)
        neigh_next = next_pot[neigh_safe]                               # (G, K)
        neigh_next = torch.where(
            valid_mask, neigh_next, torch.full_like(neigh_next, invalid_val)
        )

        log_beta[:, t] = torch.logsumexp(neigh_next, dim=1) + log_uniform

    # ---------- Posterior (gamma) ----------
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=0, keepdim=True)

    return log_gamma.exp()

# ==========================================
# 4. Parameter Update (M-Step)
# ==========================================

def update_soft_parameters(
    features: torch.Tensor,
    old_params: Dict[str, torch.Tensor], 
    stpd_gt: torch.Tensor, 
    grid_angle_qg: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Update parameters based on Spatio-Temporal Probability Distribution (stpd).
    """
    # Hyperparameter
    MIN_SCOPE_AOA, MIN_BIAS_AOA = 0.0, 1.0

    # Clone params to avoid in-place modification issues
    new_params = {k: v.clone() for k, v in old_params.items()}
    
    # Dimensions
    Q, T, C, _ = features.shape
    G = stpd_gt.shape[0]

    # =========================================================================
    # Part 0: Smart Candidate Selection
    # Method: Select the candidate closest to Spatio-Temporal Probability Distribution's belief
    # =========================================================================
    best_features = _select_best_candidate(
        features, old_params, stpd_gt, grid_angle_qg
    ) # Returns [Q, T, 5]

    # Extract Features
    obs_aoa_main = best_features[..., 0]
    obs_aoa_sprd = best_features[..., 1]

    # =========================================================================
    # Part 1: Update Offsets
    # Method: Inverse-Variance Weighting
    # =========================================================================

    # Calculate Old Variance [Q, T]
    aoa_var = old_params['aoa_weight'] * (obs_aoa_sprd ** 2) + old_params['aoa_bias_q'].unsqueeze(1)

    # Clamp for stability [Q, T]
    aoa_var = torch.clamp(aoa_var, min=1e-6)

    # Calculate Offset Weights [Q, G, T]
    weight_aoa_offset_qgt = stpd_gt.unsqueeze(0) / aoa_var.unsqueeze(1)

    # Calculate Residuals (Obs - Geo) [Q, G, T]
    diff_aoa = obs_aoa_main.unsqueeze(1) - grid_angle_qg.unsqueeze(2)
    diff_aoa = torch.remainder(diff_aoa + 180.0, 360.0) - 180.0

    # Update Offsets [Q]
    raw_aoa_offset = torch.sum(weight_aoa_offset_qgt * diff_aoa, dim=(1, 2)) / (torch.sum(weight_aoa_offset_qgt, dim=(1, 2)) + 1e-9)
    new_params['aoa_offset_q'] = raw_aoa_offset

    # =========================================================================
    # Part 2: Update Variance Parameters
    # Method: Weighted Linear Regression
    # =========================================================================

    # Calculate Target Variance [Q, G, T]
    sq_err_aoa = (diff_aoa - new_params['aoa_offset_q'].view(Q, 1, 1)) ** 2

    # Prepare Regression Inputs
    # 1. Weights [Q, G, T]
    weights_reg = stpd_gt.unsqueeze(0).expand(Q, -1, -1)

    # 2. Input [Q, G, T]
    x_reg_aoa = (obs_aoa_sprd ** 2).unsqueeze(1).expand(-1, G, -1)

    # Execute Regression
    new_params['aoa_weight'], new_params['aoa_bias_q'] = (
        _weighted_linear_regression(
            x=x_reg_aoa, 
            y=sq_err_aoa, 
            w=weights_reg, 
            min_slope=MIN_SCOPE_AOA, 
            min_bias=MIN_BIAS_AOA
        )
    )

    return new_params

def _select_best_candidate(
    features: torch.Tensor,
    params: Dict[str, torch.Tensor],
    stpd_gt: torch.Tensor,
    grid_angle_qg: torch.Tensor
) -> torch.Tensor:
    """
    Helper: Selects the candidate closest to the Expected Ground Truth.
    """
    Q, T, C, _ = features.shape
    
    # 1. Expected Ground Truth from STPD [Q, T]
    # Norm STPD
    stpd_sum_1t = torch.sum(stpd_gt, dim=0, keepdim=True) + 1e-9
    stpd_norm_gt = stpd_gt / stpd_sum_1t

    # Expected Angle (Cyclic Mean)
    grid_angle_rad_qg = torch.deg2rad(grid_angle_qg)
    sin_grid_qg = torch.sin(grid_angle_rad_qg)
    cos_grid_qg = torch.cos(grid_angle_rad_qg)
    exp_sin_qt = torch.matmul(sin_grid_qg, stpd_norm_gt)
    exp_cos_qt = torch.matmul(cos_grid_qg, stpd_norm_gt)
    exp_aoa_rad_qt = torch.atan2(exp_sin_qt, exp_cos_qt)
    exp_aoa_qt = torch.rad2deg(exp_aoa_rad_qt)

    # 2. Compare Candidates
    # Offsets for comparison [Q, 1]
    offset_aoa_q1 = params['aoa_offset_q'].unsqueeze(1)

    best_idx = torch.zeros((Q, T), dtype=torch.long, device=features.device)
    min_err = torch.full((Q, T), float('inf'), device=features.device)

    # Precompute per-(Q,T) means with offsets
    mu_aoa_qt = exp_aoa_qt + offset_aoa_q1  # (Q,T)

    VAR_AOA_MIN = 1e-6    # deg^2

    for c in range(C):
        cand_aoa = features[:, :, c, 0]
        sprd_aoa = features[:, :, c, 1]

        diff = _angular_diff(cand_aoa, mu_aoa_qt)
        var = params['aoa_weight'] * (sprd_aoa ** 2) + params['aoa_bias_q'].unsqueeze(1)
        var = torch.clamp(var, min=VAR_AOA_MIN)

        err = (diff ** 2) / var + torch.log(var)

        mask = err < min_err
        min_err[mask] = err[mask]
        best_idx[mask] = c

    gather_idx = best_idx.view(Q, T, 1, 1).expand(-1, -1, 1, 5)
    return torch.gather(features, 2, gather_idx).squeeze(2)

def _angular_diff(a, b):
    """ Diff in range [-180, 180] """
    d = a - b
    return torch.remainder(d + 180.0, 360.0) - 180.0

def _weighted_linear_regression(x, y, w, min_slope, min_bias):
    """
    Solves w (Global) and b (Per-AP) using weighted least squares.
    """
    # Weighted Mean per AP (Q,)
    sum_w = torch.sum(w, dim=(1, 2)) + 1e-9
    mean_x = torch.sum(w * x, dim=(1, 2)) / sum_w
    mean_y = torch.sum(w * y, dim=(1, 2)) / sum_w
    
    # Centering (Broadcast means)
    # (Q, G, T) - (Q, 1, 1)
    x_centered = x - mean_x.view(-1, 1, 1)
    y_centered = y - mean_y.view(-1, 1, 1)
    
    # Global Slope (w)
    # Sum over Q, G, T
    numerator = torch.sum(w * x_centered * y_centered)
    denominator = torch.sum(w * (x_centered ** 2)) + 1e-9
    w_slope = numerator / denominator
    
    # Constraint: Positive correlation
    w_slope = torch.clamp(w_slope.view(1), min=min_slope)
    
    # Local Intercept (b) per AP
    b_intercept = mean_y - w_slope * mean_x
    b_intercept = torch.clamp(b_intercept, min=min_bias)
    
    return w_slope, b_intercept

# ==========================================
# 5. Viterbi hook: transition scoring
# ==========================================

def get_max_previous_score(
    neighbor_index_matrix: torch.Tensor, 
    delta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates best previous node.
    Hybrid Logic: AI Prediction + Geometric Constraints.
    """
    
    G = delta.shape[0]
    K = neighbor_index_matrix.shape[1]
    device = delta.device

    # --- Geometric Constraints ---
    invalid_score = -float('inf')
    score_g_k = torch.full((G, K), invalid_score, dtype=torch.float32, device=device)
    
    valid_mask = neighbor_index_matrix != -1
    delta_prev = delta

    for neighbor_pos in range(K):
        neighbor_indices = neighbor_index_matrix[:, neighbor_pos]
        mask = valid_mask[:, neighbor_pos]

        if mask.any():
            # Base Geometric Score
            geo_score = delta_prev[neighbor_indices[mask]]
            
            # Add AI score if valid
            # combined_score = geo_score + ai_scores[...]
            score_g_k[mask, neighbor_pos] = geo_score

    # --- Find Max ---
    max_vals, relative_indices = torch.max(score_g_k, dim=1)
    row_indices = torch.arange(G, device=device)
    winner_indices = neighbor_index_matrix[row_indices, relative_indices]
    
    return winner_indices, max_vals