import torch
import math
from typing import Dict, Any, Optional

from . import grid_tools

def gaussian_log_pdf(
        x: torch.Tensor, 
        mean: torch.Tensor, 
        variance: torch.Tensor, 
) -> torch.Tensor:
    """
    Calculate the Log Probability Density of a Gaussian distribution.
    """
    MIN_VAR_FOR_MODEL = 1e-4
    variance = torch.clamp(variance, min=MIN_VAR_FOR_MODEL)

    log_prob = -0.5 * torch.log(2.0 * torch.pi * variance) - 0.5 * (x - mean)**2 / variance

    return log_prob

def calculate_weighted_average(
        data: torch.Tensor, 
        weights: torch.Tensor, 
        dim: int = -1
    ) -> torch.Tensor:
    """
    Calculate the weighted average along a specific dimension.

    Args:
        data: The values to average.
        weights: The weights must have same shape as data or be broadcastable.
        dim: The dimension to reduce. Default is last dimension.
    """
    # Weighted Sum
    weighted_sum = (data * weights).sum(dim=dim)
    
    # Sum of Weights
    sum_weights = weights.sum(dim=dim)
    
    # Avoid division by zero
    sum_weights = torch.clamp(sum_weights, min=1e-10)

    return weighted_sum / sum_weights

def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the shortest difference between two angles in degrees.
    Handles the cyclic nature (e.g., diff between 359 and 1 is 2, not 358).
    
    Range: [-180, 180]
    """
    diff = pred - target
    # Map to [-180, 180]
    diff = torch.remainder(diff + 180.0, 360.0) - 180.0
    return diff

# TODO:目前還沒用到這個，proposed才會用到
def weighted_angular_average(
    angles: torch.Tensor, 
    weights: torch.Tensor, 
    dim: int = -1
) -> torch.Tensor:
    """
    Calculate the weighted average of angles.
    Standard averaging fails for angles (avg of 359 and 1 should be 0, not 180).
    We use vector components (sin/cos) to calculate this.
    """
    # Convert to radians
    angles_rad = torch.deg2rad(angles)
    
    # Decompose to components
    sin_val = torch.sin(angles_rad)
    cos_val = torch.cos(angles_rad)
    
    # Calculate weighted average of components
    avg_sin = calculate_weighted_average(sin_val, weights, dim)
    avg_cos = calculate_weighted_average(cos_val, weights, dim)
    
    # Convert back to degrees
    avg_angle_rad = torch.atan2(avg_sin, avg_cos)
    avg_angle_deg = torch.rad2deg(avg_angle_rad)
    
    # Map to [0, 360) or [-180, 180] as needed, here we prefer standard mod
    avg_angle_deg = torch.fmod(avg_angle_deg + 360.0, 360.0)
    
    return avg_angle_deg

def calculate_emission_probability(
        features: torch.Tensor, 
        reference_grid: torch.Tensor,
        propagation_params: Dict[str, Any], 
        ap_locations: torch.Tensor, 
        ap_orientations: torch.Tensor, 
        device: torch.device
) -> torch.Tensor:
    
##### --- Extract Power, Angle, Delay observations (Q, T) ---
    power_qt = features[:, :, 0] 
    angle_qt = features[:, :, 1] 
    delay_qt = features[:, :, 2] 
    
    G = reference_grid.shape[0] # Number of grid points
    Q, T = power_qt.shape       # Number of APs, Number of time steps
    K = 2                       # Number of states (LOS/NLOS)
    DEVICE = device
    
##### --- Parameter Preparation and Reshaping ---
    # (Q, K) -> (1, Q, 1, K) for broadcasting
    alpha_1q1k = propagation_params['alpha_qk'].to(DEVICE).unsqueeze(0).unsqueeze(2)
    beta_1q1k  = propagation_params['beta_qk'].to(DEVICE).unsqueeze(0).unsqueeze(2)
    power_var_1q1k = propagation_params['power_qk_var'].to(DEVICE).unsqueeze(0).unsqueeze(2)

    # (K) -> (1, 1, 1, K) for broadcasting
    angle_var_111k = propagation_params['angle_k_var'].to(DEVICE).view(1, 1, 1, K)
    delay_mean_111k = propagation_params['delay_k_mean'].to(DEVICE).view(1, 1, 1, K)
    delay_var_111k = propagation_params['delay_k_var'].to(DEVICE).view(1, 1, 1, K)
    
##### --- Grid Distance (L_gq) and Angle Mean Calculation ---
    # Calculate log10(Distance) for each grid point g to each AP q, shape (G, Q)
    L_gq = grid_tools.calculate_L_gq(reference_grid, ap_locations).to(DEVICE)
    # Reshape for broadcasting to (G, Q, 1, 1)
    L_gq11 = L_gq.unsqueeze(2).unsqueeze(3)

    # Calculate geometric angle mean for each grid point g to each AP q, shape (G, Q)
    angle_gq_mean = grid_tools.calculate_angle_gq_mean(reference_grid, ap_locations, ap_orientations).to(DEVICE)
    # Reshape for broadcasting to (G, Q, 1, 1)
    angle_gq11_mean = angle_gq_mean.unsqueeze(2).unsqueeze(3)
    
##### --- Feature Mean and Variance Calculation (Broadcasted) ---
    
    # --- Power ---
    # Power Mean (G, Q, 1, K) = beta_qk - (alpha_qk * log10(L_gq)) (Log-distance model)
    power_gq1k_mean = beta_1q1k - (alpha_1q1k * L_gq11) 
    # Power Mean (G, Q, T, K) - Expand over T dimension
    power_gqtk_mean = power_gq1k_mean.expand(G, Q, T, K) 
    # Power Var (G, Q, T, K) - Expand over T dimension
    power_gqtk_var = power_var_1q1k.expand(G, Q, T, K)

    # --- Angle ---
    # Angle Mean (G, Q, T, K) - Expand over T and K dimensions
    angle_gqtk_mean = angle_gq11_mean.expand(G, Q, T, K)
    # Angle Var (G, Q, T, K) - Expand over G, Q, T dimensions
    angle_gqtk_var = angle_var_111k.expand(G, Q, T, K)

    # --- Delay ---
    # Delay Mean (G, Q, T, K) - Expand over G, Q, T dimensions
    delay_gqtk_mean = delay_mean_111k.expand(G, Q, T, K)
    # Delay Var (G, Q, T, K) - Expand over G, Q, T dimensions
    delay_gqtk_var = delay_var_111k.expand(G, Q, T, K)

##### --- Reshape Observations for Broadcasting ---
    # Observations (Q, T) -> (1, Q, T, 1) -> (G, Q, T, K)
    power_gqtk = power_qt.unsqueeze(0).unsqueeze(3).expand(G, Q, T, K)
    angle_gqtk = angle_qt.unsqueeze(0).unsqueeze(3).expand(G, Q, T, K)
    delay_gqtk = delay_qt.unsqueeze(0).unsqueeze(3).expand(G, Q, T, K)
    
##### --- Calculate Log PDF for each feature (G, Q, T, K) ---
    log_P_power_gqtk = gaussian_log_pdf(power_gqtk, power_gqtk_mean, power_gqtk_var)
    log_P_angle_gqtk = gaussian_log_pdf(angle_gqtk, angle_gqtk_mean, angle_gqtk_var)
    log_P_delay_gqtk = gaussian_log_pdf(delay_gqtk, delay_gqtk_mean, delay_gqtk_var)

##### --- Incorporate Global LOS Prior pi_k ---
    # (K) -> (1, 1, 1, K) -> (G, Q, T, K)
    pi_k = propagation_params['pi_k'].to(DEVICE)
    log_pi_k_111k = torch.log(pi_k.clamp(min=1e-10)).view(1, 1, 1, K)
    log_pi_k_gqtk = log_pi_k_111k.expand(G, Q, T, K)

##### --- Emission Probability ---
    log_joint_prob_gqtk = log_pi_k_gqtk + log_P_power_gqtk + log_P_angle_gqtk + log_P_delay_gqtk
    log_joint_prob_gqt = torch.logsumexp(log_joint_prob_gqtk, dim=3)
    emission_log_prob_gt = log_joint_prob_gqt.sum(dim=1)

    return emission_log_prob_gt