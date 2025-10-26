import torch
from typing import Dict, Any, Optional
from torch.distributions import Normal

TypeTrajectory = torch.Tensor

def build_gaussian_distribution(mean: torch.Tensor, variance: torch.Tensor) -> Normal:
    
##### --- variance to standard deviation ---
    std_dev = torch.sqrt(variance)
    std_dev = torch.clamp(std_dev, min=1e-6)
    
##### --- Construct Normal Distribution ---
    gaussian_dist = Normal(loc=mean, scale=std_dev)
    
    return gaussian_dist

def calculate_L_tq(
        config: Dict[str, Any], 
        trajectory: TypeTrajectory, 
        Q: int
    ) -> torch.Tensor:

    ##### --- Prepare AP's position and trajectory ---
        ap_locations_list = []
        for ap_id in range(1, Q + 1):
            ap_key = f"AP_{ap_id}"
            if ap_key in config['ACCESS_POINTS']:
                ap_locations_list.append(config['ACCESS_POINTS'][ap_key]['LOCATION_M'])
            else:
                raise ValueError(f"Missing location data for AP ID {ap_id}")
            
        ap_locations = torch.tensor(ap_locations_list, dtype=torch.float32, device=trajectory.device)
        ap_locations_expanded = ap_locations.unsqueeze(1)
        trajectory_expanded = trajectory.unsqueeze(0)

    ##### --- calculate the distance for each t and q ---
        squared_diff = (ap_locations_expanded - trajectory_expanded) ** 2
        distance_matrix = torch.sqrt(squared_diff.sum(dim=2))
        distance_matrix = torch.clamp(distance_matrix, min=1e-10)
        L_qt = torch.log10(distance_matrix)

        return L_qt

def calculate_weighted_average(data_qt: torch.Tensor, 
                               gamma_qtk: torch.Tensor) -> torch.Tensor:
    
    data_qtk = data_qt.unsqueeze(2)

    weighted_sum = gamma_qtk * data_qtk

    numerator = weighted_sum.sum(dim=1)
    denominator = gamma_qtk.sum(dim=1)

    weighted_average = numerator / denominator.clamp(min=1e-10)

    return weighted_average

# ----------------------------------------------------
# --- Parameters Update ---
# ----------------------------------------------------

###########################################
##### --- Power's mean & variance --- #####
###########################################

def calculate_alpha_qk(
        power_qt: torch.Tensor, 
        power_qk_average: torch.Tensor, 
        L_qt: torch.Tensor, 
        L_qk_average: torch.Tensor, 
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:
    
    L_qtk = L_qt.unsqueeze(2)
    L_qtk_mean = L_qk_average.unsqueeze(1)
    term_L = L_qtk - L_qtk_mean

    power_qtk = power_qt.unsqueeze(2)
    power_qtk_mean = power_qk_average.unsqueeze(1)
    term_power = power_qtk - power_qtk_mean

    numerator_term = gamma_qtk * term_L * term_power
    denominator_term = gamma_qtk * (term_L ** 2)

    numerator = numerator_term.sum(dim=1)
    denominator = denominator_term.sum(dim=1)

    alpha_qk = numerator / denominator.clamp(min=1e-10)

    return alpha_qk

def calculate_beta_qk(
        alpha_qk: torch.Tensor, 
        power_qk_average: torch.Tensor, 
        L_qk_average: torch.Tensor
    ) -> torch.Tensor:
    
    term_alpha_L = alpha_qk * L_qk_average
    beta_qk = power_qk_average + term_alpha_L

    return beta_qk

def calculate_power_qtk_mean(
        alpha_qk: torch.Tensor,
        beta_qk: torch.Tensor, 
        L_qt: torch.Tensor    
    ):

    alpha_qtk = alpha_qk.unsqueeze(1)
    beta_qtk = beta_qk.unsqueeze(1)
    L_qtk = L_qt.unsqueeze(2)

    power_qtk_mean = beta_qtk - (alpha_qtk * L_qtk)

    return power_qtk_mean

def calculate_power_qk_var( 
        power_qt: torch.Tensor,
        power_qtk_mean_predicted: torch.Tensor, 
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:

    power_qtk = power_qt.unsqueeze(2)

    squared_error = (power_qtk - power_qtk_mean_predicted) ** 2
    weighted_squared_error = gamma_qtk * squared_error

    numerator = weighted_squared_error.sum(dim=1)
    denominator = gamma_qtk.sum(dim=1)

    power_qk_var = numerator / denominator.clamp(min=1e-10)
    
    return power_qk_var

###########################################
##### --- Angle's mean & variance --- #####
###########################################

def calculate_angle_qt_mean(
        config: Dict[str, Any], 
        trajectory: TypeTrajectory, 
        Q: int  
    ) -> torch.Tensor:

##### --- Prepare AP's position and trajectory ---
    ap_locations_list = []
    for ap_id in range(1, Q + 1):
        ap_key = f"AP_{ap_id}"
        if ap_key in config['ACCESS_POINTS']:
            ap_locations_list.append(config['ACCESS_POINTS'][ap_key]['LOCATION_M'])
        else:
            raise ValueError(f"Missing location data for AP ID {ap_id}")
        
    ap_locations = torch.tensor(ap_locations_list, dtype=torch.float32, device=trajectory.device)
    ap_locations_expanded = ap_locations.unsqueeze(1)
    trajectory_expanded = trajectory.unsqueeze(0)

    vector_V_qt = trajectory_expanded - ap_locations_expanded
    x_diff = vector_V_qt[..., 0]
    y_diff = vector_V_qt[..., 1]

    angle_rad_qt = torch.atan2(y_diff, x_diff)
    angle_deg_qt = torch.rad2deg(angle_rad_qt)

    angle_qt_mean = angle_deg_qt

    return angle_qt_mean

def calculate_angle_k_var(
        angle_qt_mean: torch.Tensor,
        angle_qt: torch.Tensor, 
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:
    
    angle_qtk = angle_qt.unsqueeze(2)
    angle_qtk_mean = angle_qt_mean.unsqueeze(2)

    numerator_term = gamma_qtk * ((angle_qtk - angle_qtk_mean) ** 2)
    numerator = numerator_term.sum(dim=(0, 1))

    denominator = gamma_qtk.sum(dim=(0, 1))

    angle_k_var = numerator / denominator.clamp(min=1e-10)

    return angle_k_var

###########################################
##### --- Delay's mean & variance --- #####
###########################################

def calculate_delay_k_mean(
        delay_qt: torch.Tensor, 
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:

    delay_qtk = delay_qt.unsqueeze(2)

    numerator_term = gamma_qtk * (delay_qtk)
    numerator = numerator_term.sum(dim=(0, 1))

    denominator = gamma_qtk.sum(dim=(0, 1))

    delay_k_mean = numerator / denominator.clamp(min=1e-10)

    return delay_k_mean

    pass

def calculate_delay_k_var(
        delay_k_mean: torch.Tensor, 
        delay_qt: torch.Tensor, 
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:
    
    delay_qtk = delay_qt.unsqueeze(2)
    delay_qtk_mean = delay_k_mean.view(1, 1, -1)

    squared_error = (delay_qtk - delay_qtk_mean) ** 2
    weighted_squared_error = gamma_qtk * squared_error

    numerator = weighted_squared_error.sum(dim=(0, 1))
    denominator = gamma_qtk.sum(dim=(0, 1))

    delay_k_var = numerator / denominator.clamp(min=1e-10)

    return delay_k_var

###########################################
##### --- Global & AP's LOS ratio --- #####
###########################################

def calculate_pi(
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:

    sum_gamma_over_QT = gamma_qtk.sum(dim=(0, 1))

    pi_k = sum_gamma_over_QT / (gamma_qtk.shape[0] * gamma_qtk.shape[1])

    return pi_k

def calculate_gamma_qtk(
        pi_k: torch.Tensor,                     # Shape: (K)
        power_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        angle_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        delay_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        power_qt: torch.Tensor,                 # Shape: (Q, T)
        angle_qt: torch.Tensor,                 # Shape: (Q, T)
        delay_qt: torch.Tensor,                 # Shape: (Q, T)
    ) -> torch.Tensor:

    log_P_power = power_dist.log_prob(power_qt.unsqueeze(2))
    log_P_angle = angle_dist.log_prob(angle_qt.unsqueeze(2))
    log_P_delay = delay_dist.log_prob(delay_qt.unsqueeze(2))
    log_pi_k = torch.log(pi_k.clamp(min=1e-10))

    log_unnormalized_prob_qtk = (
        log_pi_k                            
        + log_P_power
        + log_P_angle 
        + log_P_delay
    )

    unnormalized_prob_qtk = torch.exp(log_unnormalized_prob_qtk)
    normalization_constant = unnormalized_prob_qtk.sum(dim=2, keepdim=True)

    gamma_qtk = unnormalized_prob_qtk / normalization_constant.clamp(min=1e-10)

    return gamma_qtk

###########################################
##### --- EM Emission Probability --- #####
###########################################

def calculate_MEPLL_PropParams(
        pi_k: torch.Tensor,                     # Shape: (K)
        power_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        angle_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        delay_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        power_qt: torch.Tensor,                 # Shape: (Q, T)
        angle_qt: torch.Tensor,                 # Shape: (Q, T)
        delay_qt: torch.Tensor,                 # Shape: (Q, T)
        gamma_qtk: torch.Tensor                 # Shape: (Q, T, K)
    ) -> torch.Tensor:

    K = gamma_qtk.shape[2]

    log_P_power = power_dist.log_prob(power_qt.unsqueeze(2))
    log_P_angle = angle_dist.log_prob(angle_qt.unsqueeze(2))
    log_P_delay = delay_dist.log_prob(delay_qt.unsqueeze(2))
    log_pi_k = torch.log(pi_k.clamp(min=1e-10))

    log_joint_prob_qtk = (
        log_pi_k
        + log_P_power 
        + log_P_angle
        + log_P_delay
    )

    log_marginal_likelihood_qt = torch.logsumexp(log_joint_prob_qtk, dim=2)

    MEPLL_PropParams = log_marginal_likelihood_qt.sum()

    return MEPLL_PropParams