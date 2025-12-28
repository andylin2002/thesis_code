import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

from .._common import math_tools

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
    num_ap = len(config['ACCESS_POINTS'])
    Q = num_ap
    
    aoa_weight =    torch.ones(1, dtype=torch.float32, device=device)
    aoa_bias_q =    torch.zeros(Q, dtype=torch.float32, device=device)
    aoa_offset_q =  torch.zeros(Q, dtype=torch.float32, device=device)

    tof_weight =    torch.ones(1, dtype=torch.float32, device=device)
    tof_bias_q =    torch.zeros(Q, dtype=torch.float32, device=device)
    tof_offset_q =  torch.zeros(Q, dtype=torch.float32, device=device)

    return {
        'aoa_weight':   aoa_weight,
        'aoa_bias_q':   aoa_bias_q, 
        'aoa_offset_q': aoa_offset_q, 
        'tof_weight':   tof_weight, 
        'tof_bias_q':   tof_bias_q, 
        'tof_offset_q': tof_offset_q
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
    grid_expanded = reference_grid.unsqueeze(0)
    ap_locations_expanded = ap_locations.unsqueeze(1)

    vector_V_qg = grid_expanded - ap_locations_expanded
    x_diff = vector_V_qg[..., 0]
    y_diff = vector_V_qg[..., 1]

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
    LIGHT_SPEED = 299792458.0

    grid_dists_qg = torch.cdist(ap_locations, reference_grid, p=2)
    grid_delay_qg = grid_dists_qg / LIGHT_SPEED

    return grid_delay_qg

# ==========================================
# 2. Calculate Emission log Probability Distribution (EPD)
# ==========================================

def calculate_emission_log_probs(
    features: torch.Tensor,
    params: Dict[str, torch.Tensor],
    grid_angle_qg: torch.Tensor,
    grid_delay_qg: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates Log Emission Probability: log P(Observation_t | Grid_g)
    """
    # Dimensions
    Q, T, C, _ = features.shape
    G = grid_angle_qg.shape[1]

    # Extract Features
    obs_aoa_cand = features[..., 0]
    obs_aoa_sprd = features[..., 1]
    obs_tof_cand = features[..., 2]
    obs_tof_sprd = features[..., 3]
    obs_gain_cand = features[..., 4]

    # =========================================================
    # Step 1: Calculate Mixture Weights (based on Gain)
    # =========================================================
    # Calculate Penalty
    max_gain_q, _ = torch.max(obs_gain_cand, dim=0, keepdim=True)
    norm_gain = obs_gain_cand / (max_gain_q + 1e-9)
    penalty_factor = 1.0 / (norm_gain + 0.1)

    # Normalize gains to probabilities: w_k = gain_k / sum(gain)
    sum_gain = torch.sum(obs_gain_cand, dim=2, keepdim=True) + 1e-9
    weights = obs_gain_cand / sum_gain
    
    # Convert to log domain for addition later: [Q, T, C]
    log_weights = torch.log(weights + 1e-9)

    # =========================================================
    # Step 2: Construct Gaussian Distributions (Variance)
    # =========================================================
    # Align Bias dims for addition: [Q] -> [Q, 1, 1]
    bias_aoa = params['aoa_bias_q'].view(Q, 1, 1)
    bias_tof = params['tof_bias_q'].view(Q, 1, 1)

    # Dynamic Variance based on Spread: [Q, T, C]
    var_aoa = (params['aoa_weight'] * (obs_aoa_sprd ** 2) + bias_aoa) * penalty_factor
    var_tof = (params['tof_weight'] * (obs_tof_sprd ** 2) + bias_tof) * penalty_factor

    # Clamp for stability
    var_aoa = torch.clamp(var_aoa, min=1e-6)
    var_tof = torch.clamp(var_tof, min=1e-18)

    # Reshape for Broadcasting: [Q, T, C] -> [Q, 1, T, C]
    var_aoa = var_aoa.unsqueeze(1)
    var_tof = var_tof.unsqueeze(1)

    # =========================================================
    # Step 3: Construct Mean Vectors (Grid Geometry)
    # =========================================================
    # Align Grid dims: [Q, G] -> [Q, G, 1, 1]
    grid_angle_exp = grid_angle_qg.view(Q, G, 1, 1)
    grid_delay_exp = grid_delay_qg.view(Q, G, 1, 1)

    # Align Offset dims: [Q] -> [Q, 1, 1, 1]
    offset_aoa = params['aoa_offset_q'].view(Q, 1, 1, 1)
    offset_tof = params['tof_offset_q'].view(Q, 1, 1, 1)

    # Predicted Mean: [Q, G, 1, 1]
    mean_aoa = grid_angle_exp + offset_aoa
    mean_tof = grid_delay_exp + offset_tof

    # =========================================================
    # Step 4: Calculate Mixture Log Probability
    # =========================================================
    # Reshape Obs for Broadcasting: [Q, T, C] -> [Q, 1, T, C]
    aoa_x = obs_aoa_cand.unsqueeze(1)
    tof_x = obs_tof_cand.unsqueeze(1)

    # Compute Gaussian Log PDF -> Output: [Q, G, T, C]
    # (Broadcasting handles Q, G, T, C alignment automatically)
    log_prob_aoa = _gaussian_log_pdf_angular(aoa_x, mean_aoa, var_aoa)
    log_prob_tof = _gaussian_log_pdf_linear(tof_x, mean_tof, var_tof)

    # Add Mixture Weights: log(P) + log(w)
    # log_weights: [Q, T, C] -> [Q, 1, T, C]
    total_log_prob_k = log_prob_aoa + log_prob_tof + log_weights.unsqueeze(1)

    # =========================================================
    # Step 5: Integration (LogSumExp)
    # =========================================================
    # 1. Integrate Candidates (dim=3): log(sum(exp(P_k))) -> [Q, G, T]
    mixed_log_prob = torch.logsumexp(total_log_prob_k, dim=-1)

    # 2. Integrate APs (dim=0): Sum log probs -> [G, T]
    emission_log_probs_gt = torch.sum(mixed_log_prob, dim=0)
    
    return emission_log_probs_gt

def _gaussian_log_pdf_angular(x, mean, var):
    """ Angular Gaussian Log PDF handling cyclic wrapping (-180 to 180) """
    diff = x - mean
    # Map difference to [-180, 180]
    diff = torch.remainder(diff + 180.0, 360.0) - 180.0
    return -0.5 * (torch.log(2 * torch.pi * var) + diff**2 / var)

def _gaussian_log_pdf_linear(x, mean, var):
    """ Standard Gaussian Log PDF: -0.5*log(2pi*var) - (x-mu)^2/(2var) """
    return -0.5 * (torch.log(2 * torch.pi * var) + (x - mean)**2 / var)

# ==========================================
# 2. Forward-Backward Algorithm
# ==========================================
# TODO: check
def run_forward_backward(
    emission_log_probs: torch.Tensor,     # Shape: (G, T)
    neighbor_index_matrix: torch.Tensor,  # Shape: (G, 9)
    device: torch.device
) -> torch.Tensor:
    """
    Computes Posterior (Gamma) using Forward-Backward in log-space.
    Returns: gamma (G, T) where values sum to 1 per time step.
    """
    G, T = emission_log_probs.shape
    
    # --- 1. Forward Pass (Alpha) ---
    log_alpha = torch.zeros((G, T), device=device)
    log_alpha[:, 0] = emission_log_probs[:, 0] 

    invalid_val = -1e9 # Mask value for invalid neighbors

    for t in range(1, T):
        prev_alpha = log_alpha[:, t-1]
        
        # Gather neighbor alphas: (G, 9)
        neighbor_alphas = prev_alpha[neighbor_index_matrix] 
        
        # Mask invalid neighbors
        mask = (neighbor_index_matrix == -1)
        neighbor_alphas[mask] = invalid_val
        
        # LogSumExp (assume uniform transition probability)
        log_transition_sum = torch.logsumexp(neighbor_alphas, dim=1)
        
        log_alpha[:, t] = emission_log_probs[:, t] + log_transition_sum

    # --- 2. Backward Pass (Beta) ---
    log_beta = torch.zeros((G, T), device=device)
    
    for t in range(T-2, -1, -1):
        next_beta = log_beta[:, t+1]
        next_emission = emission_log_probs[:, t+1]
        
        # Potential from future
        next_potential = next_beta + next_emission
        
        # Scatter back to neighbors (Symmetric Graph)
        neighbor_potentials = next_potential[neighbor_index_matrix]
        
        mask = (neighbor_index_matrix == -1)
        neighbor_potentials[mask] = invalid_val
        
        log_beta[:, t] = torch.logsumexp(neighbor_potentials, dim=1)

    # --- 3. Compute Posterior (Gamma) ---
    log_gamma_unnormalized = log_alpha + log_beta
    
    # Normalize per time step
    log_normalization = torch.logsumexp(log_gamma_unnormalized, dim=0, keepdim=True)
    log_gamma = log_gamma_unnormalized - log_normalization
    
    return log_gamma.exp()

# ==========================================
# 3. Parameter Update (M-Step)
# ==========================================

def update_soft_parameters(
    features: torch.Tensor,
    old_params: Dict[str, torch.Tensor], 
    gamma_gt: torch.Tensor, 
    grid_angle_qg: torch.Tensor, 
    grid_delay_qg: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Update parameters based on Spatio-Temporal Probability Distribution (Gamma).
    """
    # Hyperparameter
    MIN_SCOPE_AOA, MIN_BIAS_AOA = 0.0, 1.0
    MIN_SCOPE_TOF, MIN_BIAS_TOF = 0.0, 0.01
    SCALE_NS, SCALE_SQ = 1e9, 1e18

    # Clone params to avoid in-place modification issues
    new_params = {k: v.clone() for k, v in old_params.items()}
    
    # Dimensions
    Q, T, C, _ = features.shape
    G = gamma_gt.shape[0]

    # =========================================================================
    # Part 0: Smart Candidate Selection
    # Method: Select the candidate closest to Spatio-Temporal Probability Distribution's belief
    # =========================================================================
    best_features = _select_best_candidate(
        features, old_params, gamma_gt, grid_angle_qg, grid_delay_qg
    ) # Returns [Q, T, 5]

    # Extract Features
    obs_aoa_main = best_features[..., 0]
    obs_aoa_sprd = best_features[..., 1]
    obs_tof_main = best_features[..., 2]
    obs_tof_sprd = best_features[..., 3]

    # =========================================================================
    # Part 1: Update Offsets
    # Method: Inverse-Variance Weighting
    # =========================================================================

    # Calculate Old Variance [Q, T]
    aoa_var = old_params['aoa_weight'] * (obs_aoa_sprd ** 2) + old_params['aoa_bias_q'].unsqueeze(1)
    tof_var = old_params['tof_weight'] * (obs_tof_sprd ** 2) + old_params['tof_bias_q'].unsqueeze(1)

    # Clamp for stability [Q, T]
    aoa_var = torch.clamp(aoa_var, min=1e-6)
    tof_var = torch.clamp(tof_var, min=1e-6)

    # Calculate Offset Weights [Q, G, T]
    weight_aoa_offset_qgt = gamma_gt.unsqueeze(0) / aoa_var.unsqueeze(1)
    weight_tof_offset_qgt = gamma_gt.unsqueeze(0) / tof_var.unsqueeze(1)

    # Calculate Residuals (Obs - Geo) [Q, G, T]
    diff_aoa = obs_aoa_main.unsqueeze(1) - grid_angle_qg.unsqueeze(2)
    diff_aoa = torch.remainder(diff_aoa + 180.0, 360.0) - 180.0

    diff_tof = obs_tof_main.unsqueeze(1) - grid_delay_qg.unsqueeze(2)

    # Update Offsets [Q]
    momentum = 0.99
    raw_aoa_offset = torch.sum(weight_aoa_offset_qgt * diff_aoa, dim=(1, 2)) / (torch.sum(weight_aoa_offset_qgt, dim=(1, 2)) + 1e-9)
    raw_tof_offset_ns = torch.sum(weight_tof_offset_qgt * diff_tof, dim=(1, 2)) / (torch.sum(weight_tof_offset_qgt, dim=(1, 2)) + 1e-9)

    new_params['aoa_offset_q'] = momentum * old_params['aoa_offset_q'] + (1 - momentum) * raw_aoa_offset
    raw_tof_offset = raw_tof_offset_ns / SCALE_NS
    new_params['tof_offset_q'] = momentum * old_params['tof_offset_q'] + (1 - momentum) * raw_tof_offset

    # =========================================================================
    # Part 2: Update Variance Parameters
    # Method: Weighted Linear Regression
    # =========================================================================

    # Calculate Target Variance [Q, G, T]
    sq_err_aoa = (diff_aoa - new_params['aoa_offset_q'].view(Q, 1, 1)) ** 2
    sq_err_tof_ns = ((diff_tof - new_params['tof_offset_q'].view(Q, 1, 1)) ** 2) * SCALE_SQ

    # Prepare Regression Inputs
    # 1. Weights [Q, G, T]
    weights_reg = gamma_gt.unsqueeze(0).expand(Q, -1, -1)

    # 2. Input [Q, G, T]
    x_reg_aoa = (obs_aoa_sprd ** 2).unsqueeze(1).expand(-1, G, -1)
    x_reg_tof_ns = ((obs_tof_sprd * SCALE_NS) ** 2).unsqueeze(1).expand(-1, G, -1)

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

    weight_tof_ns, bias_tof_ns = (
        _weighted_linear_regression(
            x=x_reg_tof_ns, 
            y=sq_err_tof_ns, 
            w=weights_reg, 
            min_slope=MIN_SCOPE_TOF, 
            min_bias=MIN_BIAS_TOF
        )
    )

    new_params['tof_weight'] = weight_tof_ns
    new_params['tof_bias_q'] = bias_tof_ns / SCALE_SQ

    return new_params

def _select_best_candidate(
    features: torch.Tensor,
    params: Dict[str, torch.Tensor],
    gamma_gt: torch.Tensor,
    grid_angle_qg: torch.Tensor,
    grid_delay_qg: torch.Tensor
) -> torch.Tensor:
    """
    Helper: Selects the candidate closest to the Expected Ground Truth.
    """
    Q, T, C, _ = features.shape
    
    # 1. Expected Ground Truth from Gamma [Q, T]
    # Norm Gamma
    gamma_sum_1t = torch.sum(gamma_gt, dim=0, keepdim=True) + 1e-9
    gamma_norm_gt = gamma_gt / gamma_sum_1t

    # Expected Angle (Cyclic Mean)
    grid_angle_rad_qg = torch.deg2rad(grid_angle_qg)
    sin_grid_qg = torch.sin(grid_angle_rad_qg)
    cos_grid_qg = torch.cos(grid_angle_rad_qg)
    exp_sin_qt = torch.matmul(sin_grid_qg, gamma_norm_gt)
    exp_cos_qt = torch.matmul(cos_grid_qg, gamma_norm_gt)
    exp_aoa_rad_qt = torch.atan2(exp_sin_qt, exp_cos_qt)
    exp_aoa_qt = torch.rad2deg(exp_aoa_rad_qt)

    # Expected Delay
    exp_tof_qt = torch.matmul(grid_delay_qg, gamma_norm_gt)

    # 2. Compare Candidates
    # Offsets for comparison [Q, 1]
    offset_aoa_q1 = params['aoa_offset_q'].unsqueeze(1)
    offset_tof_q1 = params['tof_offset_q'].unsqueeze(1)

    best_idx = torch.zeros((Q, T), dtype=torch.long, device=features.device)
    min_err = torch.full((Q, T), float('inf'), device=features.device)

    for c in range(C):
        cand_aoa = features[:, :, c, 0]
        cand_tof = features[:, :, c, 2]

        # Residuals
        diff_aoa = _angular_diff(cand_aoa, exp_aoa_qt + offset_aoa_q1)
        diff_tof = cand_tof - (exp_tof_qt + offset_tof_q1)

        # Total Error (Approx. 1 deg ~= 1 ns weight)
        err = torch.abs(diff_aoa) + torch.abs(diff_tof * 1e9)
        
        # Update Min
        mask = err < min_err
        min_err[mask] = err[mask]
        best_idx[mask] = c

    # 3. Gather Best Features [Q, T, 5]
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
# 4. Get Viterbi algorithm Winner Information (For Estimator)
# ==========================================

def get_max_previous_score(
    neighbor_index_matrix: torch.Tensor, 
    delta: torch.Tensor, 
    t: int, 
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates best previous node.
    Hybrid Logic: AI Prediction + Geometric Constraints.
    """
    model = kwargs.get('model')
    
    G = delta.shape[0]
    device = delta.device
    
    # --- AI Prediction ---
    ai_scores = 0.0
    if model is not None:
        # TODO: Implement Transformer inference
        pass

    # --- Geometric Constraints ---
    invalid_score = -float('inf')
    score_g_9 = torch.full((G, 9), invalid_score, dtype=torch.float32, device=device)
    
    valid_mask = neighbor_index_matrix != -1
    delta_prev = delta

    for neighbor_pos in range(9):
        neighbor_indices = neighbor_index_matrix[:, neighbor_pos]
        mask = valid_mask[:, neighbor_pos]

        if mask.any():
            # Base Geometric Score
            geo_score = delta_prev[neighbor_indices[mask]]
            
            # Add AI score if valid
            # combined_score = geo_score + ai_scores[...]
            score_g_9[mask, neighbor_pos] = geo_score

    # --- Find Max ---
    max_vals, relative_indices = torch.max(score_g_9, dim=1)
    row_indices = torch.arange(G, device=device)
    winner_indices = neighbor_index_matrix[row_indices, relative_indices]
    
    return winner_indices, max_vals