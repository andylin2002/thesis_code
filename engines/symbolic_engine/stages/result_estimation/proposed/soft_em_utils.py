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

def calculate_log_pi_qtc(
    features: torch.Tensor,
    tof_params: Dict[str, float],
) -> torch.Tensor:
    """
    Compute fixed path-level log mixture weights log pi_{q,t,c}
    from gain and ToF features.
    """
    obs_tof_cand = features[..., 2]
    obs_tof_sprd = features[..., 3]
    obs_gain_cand = features[..., 4]

    log_rel_gain = _calculate_log_gain_reliability(obs_gain_cand)
    log_rel_tof = _calculate_log_tof_reliability(
        obs_tof_cand,
        obs_tof_sprd,
        tof_params,
    )

    return torch.log_softmax(log_rel_gain + log_rel_tof, dim=-1)

def calculate_emission_log_probs(
    config: Dict[str, Any], 
    features: torch.Tensor,
    params: Dict[str, torch.Tensor],
    grid_angle_qg: torch.Tensor, 
    log_pi_qtc: torch.Tensor
) -> torch.Tensor:
    """
    Calculates AoA-based emission log-probabilities.
    """
    Q, T, C, _ = features.shape
    G = grid_angle_qg.shape[1]

    # 1. Feature Extraction & Weighting
    obs_aoa_cand = features[..., 0]
    obs_aoa_sprd = features[..., 1]

    # 2. Gaussian Parameters (Variance & Mean)
    # Variance dynamic adjustment
    bias_aoa = params['aoa_bias_q'].view(Q, 1, 1)
    var_aoa = (params['aoa_weight'] * (obs_aoa_sprd ** 2) + bias_aoa).clamp(min=1e-6).unsqueeze(1)

    # Mean vectors from grid
    mean_aoa = grid_angle_qg.view(Q, G, 1, 1) + params['aoa_offset_q'].view(Q, 1, 1, 1)

    # 3. Calculate Log Probability (Per AP)
    # PDF: (Q, G, T, C)
    log_prob_aoa = _gaussian_log_pdf_angular(obs_aoa_cand.unsqueeze(1), mean_aoa, var_aoa)
    
    # Mixture Integration (LogSumExp over Candidates)
    enable_tof_gain_weight = config.get("ENABLE_TOF_GAIN_WEIGHT", True)
    if enable_tof_gain_weight:
        emission_log_probs_qgt = torch.logsumexp(log_prob_aoa + log_pi_qtc.unsqueeze(1), dim=-1)
    else:
        emission_log_probs_qgt = torch.logsumexp(log_prob_aoa, dim=-1)

    return emission_log_probs_qgt

def _gaussian_log_pdf_angular(x, mean, var):
    """ Angular Gaussian Log PDF handling cyclic wrapping (-180 to 180) """
    diff = x - mean
    # Map difference to [-180, 180]
    diff = torch.remainder(diff + 180.0, 360.0) - 180.0
    return -0.5 * (torch.log(2 * torch.pi * var) + diff**2 / var)

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
    config: Dict[str, Any],
    features: torch.Tensor,
    old_params: Dict[str, torch.Tensor],
    stpd_gt: torch.Tensor,
    grid_angle_qg: torch.Tensor,
    log_pi_qtc: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Update calibration parameters using zeta-weighted M-step.

    zeta_{q,g,t,c}
        =
        gamma_{g,t}
        * pi_{q,t,c} p_{q,g,t,c}
          / sum_j pi_{q,t,j} p_{q,g,t,j}
    """
    MIN_SCOPE_AOA = 0.0
    MIN_BIAS_AOA = 1.0

    Q, T, C, _ = features.shape
    G = stpd_gt.shape[0]

    if grid_angle_qg.shape != (Q, G):
        raise ValueError(
            f"grid_angle_qg shape mismatch: expected {(Q, G)}, "
            f"got {tuple(grid_angle_qg.shape)}"
        )

    if stpd_gt.shape != (G, T):
        raise ValueError(
            f"stpd_gt shape mismatch: expected {(G, T)}, "
            f"got {tuple(stpd_gt.shape)}"
        )

    if log_pi_qtc.shape != (Q, T, C):
        raise ValueError(
            f"log_pi_qtc shape mismatch: expected {(Q, T, C)}, "
            f"got {tuple(log_pi_qtc.shape)}"
        )

    obs_aoa_cand = features[..., 0]   # (Q, T, C)
    obs_aoa_sprd = features[..., 1]   # (Q, T, C)

    # Current variance: sigma^2 = w * spread^2 + b_q
    base_var_qtc = (
        old_params["aoa_weight"] * (obs_aoa_sprd ** 2)
        + old_params["aoa_bias_q"].view(Q, 1, 1)
    )
    base_var_qtc = torch.clamp(base_var_qtc, min=1e-6)

    # Current AoA likelihood p_{q,g,t,c}
    mean_qg11 = (
        grid_angle_qg.view(Q, G, 1, 1)
        + old_params["aoa_offset_q"].view(Q, 1, 1, 1)
    )

    obs_q1tc = obs_aoa_cand.unsqueeze(1)       # (Q, 1, T, C)
    var_q1tc = base_var_qtc.unsqueeze(1)       # (Q, 1, T, C)

    diff_old_qgtc = obs_q1tc - mean_qg11
    diff_old_qgtc = torch.remainder(diff_old_qgtc + 180.0, 360.0) - 180.0

    log_p_qgtc = -0.5 * (
        torch.log(2.0 * math.pi * var_q1tc)
        + (diff_old_qgtc ** 2) / var_q1tc
    )

    # Candidate fraction:
    # pi_{q,t,c} p_{q,g,t,c} / sum_j pi_{q,t,j} p_{q,g,t,j}
    enable_tof_gain_weight = config.get("ENABLE_TOF_GAIN_WEIGHT", True)
    if enable_tof_gain_weight:
        log_joint_qgtc = log_pi_qtc.unsqueeze(1) + log_p_qgtc
    else:
        log_joint_qgtc = log_p_qgtc  # uniform candidate weighting
        
    log_candidate_fraction_qgtc = log_joint_qgtc - torch.logsumexp(
        log_joint_qgtc,
        dim=-1,
        keepdim=True,
    )
    candidate_fraction_qgtc = torch.exp(log_candidate_fraction_qgtc)

    # Final M-step weight zeta_{q,g,t,c}
    zeta_qgtc = (
        stpd_gt.unsqueeze(0).unsqueeze(-1)
        * candidate_fraction_qgtc
    )

    # Offset update
    d_qgtc = obs_q1tc - grid_angle_qg.view(Q, G, 1, 1)
    d_qgtc = torch.remainder(d_qgtc + 180.0, 360.0) - 180.0

    offset_weight_qgtc = zeta_qgtc / torch.clamp(var_q1tc, min=1e-6)

    d_rad_qgtc = torch.deg2rad(d_qgtc)

    sin_sum_q = torch.sum(
        offset_weight_qgtc * torch.sin(d_rad_qgtc),
        dim=(1, 2, 3),
    )
    cos_sum_q = torch.sum(
        offset_weight_qgtc * torch.cos(d_rad_qgtc),
        dim=(1, 2, 3),
    )

    new_offset_q = torch.rad2deg(torch.atan2(sin_sum_q, cos_sum_q))

    # Variance mapping update
    residual_qgtc = d_qgtc - new_offset_q.view(Q, 1, 1, 1)
    residual_qgtc = torch.remainder(residual_qgtc + 180.0, 360.0) - 180.0

    x_reg_aoa = (obs_aoa_sprd ** 2).unsqueeze(1).expand(Q, G, T, C)
    y_reg_aoa = residual_qgtc ** 2

    new_weight, new_bias_q = _weighted_linear_regression(
        x=x_reg_aoa,
        y=y_reg_aoa,
        w=zeta_qgtc,
        min_slope=MIN_SCOPE_AOA,
        min_bias=MIN_BIAS_AOA,
    )

    new_params = {k: v.clone() for k, v in old_params.items()}
    new_params["aoa_offset_q"] = new_offset_q
    new_params["aoa_weight"] = new_weight
    new_params["aoa_bias_q"] = new_bias_q

    return new_params

def _weighted_linear_regression(x, y, w, min_slope, min_bias):
    """
    Solves the variance mapping parameters using weighted least squares.

    This function is used by the zeta-weighted M-step, where all tensors
    have shape (Q, G, T, C):

        y ~= slope * x + bias_q

    slope is shared across all APs, while bias_q is AP-wise.
    """
    if x.ndim != 4 or y.ndim != 4 or w.ndim != 4:
        raise ValueError(
            "_weighted_linear_regression expects 4D tensors with shape "
            "(Q, G, T, C)."
        )

    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError(
            f"Shape mismatch: x={tuple(x.shape)}, "
            f"y={tuple(y.shape)}, w={tuple(w.shape)}"
        )

    # Weighted mean per AP: sum over G, T, C
    sum_w = torch.sum(w, dim=(1, 2, 3)) + 1e-9
    mean_x = torch.sum(w * x, dim=(1, 2, 3)) / sum_w
    mean_y = torch.sum(w * y, dim=(1, 2, 3)) / sum_w

    # Centered variables per AP
    x_centered = x - mean_x.view(-1, 1, 1, 1)
    y_centered = y - mean_y.view(-1, 1, 1, 1)

    # Global slope shared across APs
    numerator = torch.sum(w * x_centered * y_centered)
    denominator = torch.sum(w * (x_centered ** 2)) + 1e-9
    w_slope = numerator / denominator

    w_slope = torch.clamp(w_slope.view(1), min=min_slope)

    # AP-wise intercept
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