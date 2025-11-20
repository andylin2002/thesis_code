import torch
from typing import Dict, Any, Optional
from torch.distributions import Normal

import utils
import numpy as np
from transformer.transformer_tool import transformer_batch_predict_logits


TypeTrajectory = torch.Tensor

def get_ap_locations(config: Dict[str, Any], Q: int, device: torch.Tensor) -> torch.Tensor:

    DEVICE = device

    ap_locations_list = []
    for ap_id in range(1, Q + 1):
        ap_key = f"AP_{ap_id}"
        if ap_key in config['ACCESS_POINTS']:
            ap_locations_list.append(config['ACCESS_POINTS'][ap_key]['LOCATION_M'])
        else:
            raise ValueError(f"Missing location data for AP ID {ap_id}")
        
    ap_locations = torch.tensor(ap_locations_list, dtype=torch.float32, device=DEVICE)

    return ap_locations


#******************************************************************************#
#******************************************************************************#
#********************************************* --- findPropParams Step --- ****#
#******************************************************************************#
#******************************************************************************#

def build_gaussian_distribution(mean: torch.Tensor, variance: torch.Tensor) -> Normal:
    
##### --- variance to standard deviation ---
    std_dev = torch.sqrt(variance)
    std_dev = torch.clamp(std_dev, min=1e-6)
    
##### --- Construct Normal Distribution ---
    gaussian_dist = Normal(loc=mean, scale=std_dev)
    
    return gaussian_dist

def calculate_L_qt(
        trajectory: TypeTrajectory, 
        ap_locations: torch.Tensor, 
    ) -> torch.Tensor:

    ##### --- Prepare AP's position and trajectory ---
        
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

def select_initial_position(
        config: Dict[str, Any], 
        reference_grid: torch.Tensor, 
        power_q: torch.Tensor, 
        device: torch.device
) -> torch.Tensor:
    
    strongest_power_ap_idx = torch.argmax(power_q).item() + 1
    ap_key = f'AP_{strongest_power_ap_idx}'

    strongest_power_ap_pos = config['ACCESS_POINTS'][ap_key]['LOCATION_M']

    if not isinstance(strongest_power_ap_pos, torch.Tensor):
        strongest_power_ap_pos = torch.tensor(strongest_power_ap_pos, dtype=torch.float32, device=device)

    distances = torch.linalg.norm(reference_grid - strongest_power_ap_pos.unsqueeze(0).to(reference_grid.device), dim=1)
    start_grid_idx = torch.argmin(distances).item()

    return reference_grid[start_grid_idx].unsqueeze(0)

from sklearn.mixture import GaussianMixture
def estimate_two_gaussians(data_flat: torch.Tensor):

    K = 2
    length = len(data_flat)
    min_length = length / 2
    device = data_flat.device

    if data_flat.is_cuda:
        data_np = data_flat.cpu().numpy()
    else:
        data_np = data_flat.numpy()

    # Remove invalid element
    data_np = data_np.astype(float)
    data_np[data_np == -90.0] = np.nan
    data_np = data_np[~np.isnan(data_np)]

    if data_np.size < min_length:
        mean = np.mean(data_np) if data_np.size > 0 else 0.0
        var = np.var(data_np) if data_np.size > 0 else 1.0
        means = np.array([mean, mean])
        vars = np.array([var, var])
        return torch.tensor(means, dtype=torch.float32, device=device), \
               torch.tensor(vars, dtype=torch.float32, device=device)
    
    data_np = data_np.reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type='spherical',
            max_iter=500,
            tol=1e-6, 
            init_params='kmeans',
            n_init=5, 
            reg_covar=1e-6
        )
        gmm.fit(data_np)

        means = gmm.means_.flatten()
        vars = gmm.covariances_.flatten()

        # sort small mean first
        idx = np.argsort(vars)
        means = means[idx]
        vars = vars[idx]

    except Exception as e:
        mean = np.mean(data_np)
        var = np.var(data_np)
        means = np.array([mean, mean])
        vars = np.array([var, var])

    # return to tensor on original device
    return torch.tensor(means, dtype=torch.float32, device=device), \
           torch.tensor(vars, dtype=torch.float32, device=device)

###########################################
##### --- Power's mean & variance --- #####
###########################################

def calculate_init_alpha_and_beta_qk(
        reference_grid: torch.Tensor, 
        ap_locations: torch.Tensor, 
        power_qt: torch.Tensor, 
    ) -> torch.Tensor:

    K = 2

    ap_locations_expanded = ap_locations.unsqueeze(1)                           # shape: [Q, 2] -> [Q, 1, 2]
    reference_grid_expanded = reference_grid.unsqueeze(0)                       # shape: [G, 2] -> [1, G, 2]

    diff = ap_locations_expanded - reference_grid_expanded                      # shape: [Q, G, 2]
    distance_square = torch.sum(diff**2, dim=2)                                 # shape: [Q, G]
    distance = torch.sqrt(distance_square)                                      # shape: [Q, G]
    max_distance = torch.max(distance, dim=1).values                            # shape: [Q]
    log_max_distance = torch.log10(max_distance)                                # shape: [Q]

    max_power_q = torch.max(power_qt, dim=1).values                             # shape: [Q]
    min_power_q = torch.min(power_qt, dim=1).values                             # shape: [Q]

    beta_q = max_power_q
    alpha_q = (beta_q - min_power_q) / log_max_distance

    beta_q1 = beta_q.unsqueeze(1)                                               # shape: [Q, 1]
    alpha_q1 = alpha_q.unsqueeze(1)                                             # shape: [Q, 1]

    init_beta_qk = beta_q1.repeat(1, K)                                         # shape: [Q, K]
    init_alpha_qk = alpha_q1.repeat(1, K)                                       # shape: [Q, K]

    return init_alpha_qk, init_beta_qk

def calculate_init_power_qk_var(
        reference_grid: torch.Tensor, 
        ap_locations: torch.Tensor, 
        alpha_qk: torch.Tensor, 
        beta_qk: torch.Tensor, 
        power_qt: torch.Tensor
    ) -> torch.Tensor:

    K = 2

    L_gq = calculate_L_gq(reference_grid, ap_locations)

    alpha_q = alpha_qk[:, 0]
    beta_q = beta_qk[:, 0]
    alpha_1q = alpha_q.unsqueeze(0)
    beta_1q = beta_q.unsqueeze(0)

    power_gq_mean = beta_1q - (alpha_1q * L_gq)
    power_gq1_mean = power_gq_mean.unsqueeze(2)

    power_1qt = power_qt.unsqueeze(0)

    error_gqt = abs(power_1qt - power_gq1_mean)
    min_error_qt = torch.min(error_gqt, dim=0).values  # find the most probable point(g) for each q and t
    squared_min_error_qt = min_error_qt**2

    power_q_var = torch.mean(squared_min_error_qt, dim=1)
    power_q1_var = power_q_var.unsqueeze(1)
    init_power_qk_var = power_q1_var.repeat(1, K)

    return init_power_qk_var

def calculate_init_angle_k_var( # FIXME
        reference_grid: torch.Tensor, 
        ap_locations: torch.Tensor,
        ap_orientations: torch.Tensor, 
        angle_qt: torch.Tensor, 
    ) -> torch.Tensor:

    K = 2

    angle_gq_mean = calculate_angle_gq_mean(reference_grid, ap_locations, ap_orientations)
    angle_gq1_mean = angle_gq_mean.unsqueeze(2)

    angle_1qt = angle_qt.unsqueeze(0)

    error_gqt = abs(angle_1qt - angle_gq1_mean)
    min_error_qt = torch.min(error_gqt, dim=0).values  # find the most probable point(g) for each q and t
    squared_min_error_qt = min_error_qt**2

    angle_var = torch.mean(squared_min_error_qt)
    angle_k_var = angle_var.repeat(K)

    return angle_k_var

def calculate_alpha_qk(
        power_qt: torch.Tensor, 
        power_qk_average: torch.Tensor, 
        L_qt: torch.Tensor, 
        L_qk_average: torch.Tensor, 
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:
    
    L_qtk = L_qt.unsqueeze(2)
    L_qtk_average = L_qk_average.unsqueeze(1)
    term_L = L_qtk - L_qtk_average

    power_qtk = power_qt.unsqueeze(2)
    power_qtk_average = power_qk_average.unsqueeze(1)
    term_power = power_qtk - power_qtk_average

    numerator_term = gamma_qtk * term_L * term_power
    denominator_term = gamma_qtk * (term_L ** 2)

    numerator = numerator_term.sum(dim=1)
    denominator = denominator_term.sum(dim=1)
    
    alpha_qk = -1 * numerator / (denominator.clamp(min=1e-10))

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

    alpha_q1k = alpha_qk.unsqueeze(1)
    beta_q1k = beta_qk.unsqueeze(1)
    L_qt1 = L_qt.unsqueeze(2)

    power_qtk_mean = beta_q1k - (alpha_q1k * L_qt1)

    return power_qtk_mean

def calculate_power_qk_var( 
        power_qt: torch.Tensor,
        power_qtk_mean_predicted: torch.Tensor, 
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:

    power_qt1 = power_qt.unsqueeze(2)

    squared_error = (power_qt1 - power_qtk_mean_predicted) ** 2
    weighted_squared_error = gamma_qtk * squared_error

    numerator = weighted_squared_error.sum(dim=1)
    denominator = gamma_qtk.sum(dim=1)

    power_qk_var = numerator / denominator.clamp(min=1e-10)
    
    return power_qk_var + 1e-6

###########################################
##### --- Angle's mean & variance --- #####
###########################################

def calculate_angle_qt_mean(
        trajectory: TypeTrajectory, 
        ap_locations: torch.Tensor,
        ap_orientations: torch.Tensor
    ) -> torch.Tensor:

    ap_locations_expanded = ap_locations.unsqueeze(1) # shape: [Q, (T), 2]
    trajectory_expanded = trajectory.unsqueeze(0) # shape: [(Q), T, 2]

    vector_V_qt = trajectory_expanded - ap_locations_expanded
    x_diff = vector_V_qt[..., 0]
    y_diff = vector_V_qt[..., 1]

    angle_rad_qt = torch.atan2(y_diff, x_diff)
    angle_deg_qt = torch.rad2deg(angle_rad_qt)
    angle_deg_qt = torch.fmod(torch.fmod(angle_deg_qt, 360.0) + 360.0, 360.0)

    ap_orientations_expanded = ap_orientations.unsqueeze(1).expand_as(angle_deg_qt) # shape: [Q, T]
    relative_angle_deg = ap_orientations_expanded - angle_deg_qt

    angle_qt_mean = torch.clamp(relative_angle_deg, min=-90.0, max=90.0)

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

    return angle_k_var + 1e-6

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

    return delay_k_var + 1e-6

###########################################
##### --- Global & AP's LOS ratio --- #####
###########################################

def calculate_pi(
        gamma_qtk: torch.Tensor
    ) -> torch.Tensor:

    sum_gamma_over_QT = gamma_qtk.sum(dim=(0, 1))

    pi_k = sum_gamma_over_QT / (gamma_qtk.shape[0] * gamma_qtk.shape[1])

    return pi_k

def calculate_init_gamma_qtk(
        delay_dist: torch.distributions.Normal,
        delay_qt: torch.Tensor
    ) -> torch.Tensor:

    log_P_delay = delay_dist.log_prob(delay_qt.unsqueeze(2))

    log_unnormalized_prob_qtk = log_P_delay
    unnormalized_prob_qtk = torch.exp(log_unnormalized_prob_qtk)
    normalization_constant = unnormalized_prob_qtk.sum(dim=2, keepdim=True)

    init_gamma_qtk = unnormalized_prob_qtk / normalization_constant.clamp(min=1e-10)

    return init_gamma_qtk

def calculate_gamma_qtk(
        pi_k: torch.Tensor,                     # Shape: (K)
        power_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        angle_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        delay_dist: torch.distributions.Normal, # Batch Shape: (Q, T, K)
        power_qt: torch.Tensor,                 # Shape: (Q, T)
        angle_qt: torch.Tensor,                 # Shape: (Q, T)
        delay_qt: torch.Tensor                  # Shape: (Q, T)
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
    ) -> torch.Tensor:

    log_P_power_qtk = power_dist.log_prob(power_qt.unsqueeze(2))
    log_P_angle_qtk = angle_dist.log_prob(angle_qt.unsqueeze(2))
    log_P_delay_qtk = delay_dist.log_prob(delay_qt.unsqueeze(2))
    log_pi_k = torch.log(pi_k.clamp(min=1e-10))

    log_joint_prob_qtk = (
        log_pi_k
        + log_P_power_qtk 
        + log_P_angle_qtk
        + log_P_delay_qtk
    )

    DEBUG = True
    if DEBUG:
        q = 0
        t = 0
        print("log_pi_k: ", log_pi_k)
        print("log_P_power: ", log_P_power_qtk[q][t])
        print("log_P_angle: ", log_P_angle_qtk[q][t])
        print("log_P_delay: ", log_P_delay_qtk[q][t])

    log_marginal_likelihood_qt = torch.logsumexp(log_joint_prob_qtk, dim=2)

    MEPLL_PropParams = log_marginal_likelihood_qt.sum()

    return MEPLL_PropParams

#******************************************************************************#
#******************************************************************************#
#********************************************* --- findTrajectory Step --- ****#
#******************************************************************************#
#******************************************************************************#

import torch
from typing import Dict, Any

def get_all_neighbor_indices(
    config: Dict[str, Any],
    grid_indices_G: torch.Tensor, 
    device: torch.device
) -> torch.Tensor:
    
    # Parameter Setup
    W = config['X_WIDTH'] 
    H = config['Y_WIDTH'] 
    G = grid_indices_G.shape[0] # Numbers of Reference Point
    
    # grid_indices_G must be of long type
    grid_indices_G = grid_indices_G.long() 
    
    # [G] -> [G, 1]
    current_rows_G = (grid_indices_G // W).unsqueeze(1) 
    current_cols_G = (grid_indices_G % W).unsqueeze(1)
    
    # Offset matrix (3x3 = 9 combinations), shape [1, 9]
    offset_row_1_9 = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.long, device=device).unsqueeze(0)
    offset_col_1_9 = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.long, device=device).unsqueeze(0)
  
    # Broadcasting [G, 1] + [1, 9] -> [G, 9]
    candidate_rows_G_9 = current_rows_G + offset_row_1_9
    candidate_cols_G_9 = current_cols_G + offset_col_1_9
    
    # Boundary Check (using mask), shape [G, 9]
    mask_row_valid = (candidate_rows_G_9 >= 0) & (candidate_rows_G_9 < H)
    mask_col_valid = (candidate_cols_G_9 >= 0) & (candidate_cols_G_9 < W)
    valid_mask_G_9 = mask_row_valid & mask_col_valid
    
    # Convert 2D candidate coordinates to 1D indices, applying the validity mask
    all_candidate_indices_G_9 = candidate_rows_G_9 * W + candidate_cols_G_9
    neighbor_indices = torch.full((G, 9), -1, dtype=torch.long, device=device)
    neighbor_indices[valid_mask_G_9] = all_candidate_indices_G_9[valid_mask_G_9]
    
    return neighbor_indices

###########################################
##### --- Emission Probability GT --- #####
###########################################

def calculate_emission_probability(
        feature_matrix: torch.Tensor, 
        reference_grid: torch.Tensor,
        propagation_params: Dict[str, Any], 
        ap_locations: torch.Tensor, 
        ap_orientations: torch.Tensor, 
        device: torch.device
) -> torch.Tensor:
    
##### --- Extract Power, Angle, Delay observations (Q, T) ---
    power_qt = feature_matrix[:, :, 0] 
    angle_qt = feature_matrix[:, :, 1] 
    delay_qt = feature_matrix[:, :, 2] 
    
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
    L_gq = calculate_L_gq(reference_grid, ap_locations).to(DEVICE)
    # Reshape for broadcasting to (G, Q, 1, 1)
    L_gq11 = L_gq.unsqueeze(2).unsqueeze(3)

    # Calculate geometric angle mean for each grid point g to each AP q, shape (G, Q)
    angle_gq_mean = calculate_angle_gq_mean(reference_grid, ap_locations, ap_orientations).to(DEVICE)
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

    DEBUG = False
    if DEBUG:
        print("**** point47 AP2 t=0 ****")
        print("power_gqtk: ", power_gqtk[47, 1, 0]), 
        print("power_gqtk_mean: ", power_gqtk_mean[47, 1, 0])
        print("power_gqtk_var: ", power_gqtk_var[47, 1, 0])

        print("**** point50 AP2 t=0 ****")
        print("power_gqtk: ", power_gqtk[50, 1, 0]), 
        print("power_gqtk_mean: ", power_gqtk_mean[50, 1, 0])
        print("power_gqtk_var: ", power_gqtk_var[50, 1, 0])

    DEBUG = False
    if DEBUG:
        print("########## point47 ##########")
        print("log_pi_k_gqtk: ", log_pi_k_gqtk[47, :, 0])
        print("log_P_power_gqtk: ", log_P_power_gqtk[47, :, 0])
        print("log_P_angle_gqtk: ", log_P_angle_gqtk[47, :, 0])
        print("log_P_delay_gqtk: ", log_P_delay_gqtk[47, :, 0])
        print("\n")
        print("########## point50 ##########")
        print("log_pi_k_gqtk: ", log_pi_k_gqtk[50, :, 0])
        print("log_P_power_gqtk: ", log_P_power_gqtk[50, :, 0])
        print("log_P_angle_gqtk: ", log_P_angle_gqtk[50, :, 0])
        print("log_P_delay_gqtk: ", log_P_delay_gqtk[50, :, 0])

    log_joint_prob_gqtk = log_pi_k_gqtk + log_P_power_gqtk + log_P_angle_gqtk + log_P_delay_gqtk
    log_joint_prob_gqt = torch.logsumexp(log_joint_prob_gqtk, dim=3)
    emission_log_prob_gt = log_joint_prob_gqt.sum(dim=1)

    return emission_log_prob_gt

def calculate_angle_gq_mean(
        reference_grid: torch.Tensor, # Shape [G, 2]
        ap_locations: torch.Tensor, 
        ap_orientations: torch.Tensor
) -> torch.Tensor:
    
    ap_locations_expanded = ap_locations.unsqueeze(0)     # Shape [1, Q, 2]
    grid_expanded = reference_grid.unsqueeze(1)           # Shape [G, 1, 2]

    vector_V_gq = ap_locations_expanded - grid_expanded
    x_diff = vector_V_gq[..., 0]
    y_diff = vector_V_gq[..., 1]

    angle_rad_gq = torch.atan2(y_diff, x_diff)
    angle_deg_gq = torch.rad2deg(angle_rad_gq)

    angle_deg_gq = torch.fmod(torch.fmod(angle_deg_gq, 360.0) + 360.0, 360.0)

    ap_orientations_expanded = ap_orientations.unsqueeze(0).expand_as(angle_deg_gq) # shape: [Q, T]
    relative_angle_deg = ap_orientations_expanded - angle_deg_gq

    angle_gq_mean = torch.clamp(relative_angle_deg, min=-90.0, max=90.0)

    return angle_gq_mean

def calculate_L_gq(
        reference_grid: torch.Tensor, 
        ap_locations: torch.Tensor, 
) -> torch.Tensor:
    
    ap_locations_expanded = ap_locations.unsqueeze(0)     # Shape [1, Q, 2]
    grid_expanded = reference_grid.unsqueeze(1)           # Shape [G, 1, 2]

    squared_diff = (ap_locations_expanded - grid_expanded) ** 2
    distance_matrix = torch.sqrt(squared_diff.sum(dim=2))
    distance_matrix = torch.clamp(distance_matrix, min=1e-10)
    L_gq = torch.log10(distance_matrix)

    return L_gq

def gaussian_log_pdf(
        x: torch.Tensor, 
        mean: torch.Tensor, 
        variance: torch.Tensor, 
) -> torch.Tensor:
    MIN_VAR_FOR_MODEL = 1e-4
    variance = torch.clamp(variance, min=MIN_VAR_FOR_MODEL)

    log_prob = -0.5 * torch.log(2.0 * torch.pi * variance) - 0.5 * (x - mean)**2 / variance

    return log_prob


###########################################
##### --- Ping-Pong Updating step --- #####
###########################################

def get_winner_neighbor_info( # FIXME 仔細檢查這裡每一步（如果參數更新沒錯那就是這裡錯，我發現參數才更新一次預測出來的路徑就整個不對，應該有問題）
    t: int, 
    reference_grid: torch.Tensor, 
    ref_index: int, 
    G_neighbor_index_matrix: torch.Tensor, 
    delta: torch.Tensor, 
    path: torch.Tensor, 
    model: Optional[torch.nn.Module], 
    mode: str, 
    SOS_TOKEN: int
) -> tuple[torch.Tensor, torch.Tensor]:
    
    G = delta.shape[0]

##### --- Transition Log Probability Using Transformer ---
    if model is not None:
        with torch.no_grad():

            history_coords = utils.get_history_coords_batch(reference_grid, ref_index, path)

            transition_log_prob_G_9 = transformer_batch_predict_logits(
                model, 
                history_coords, 
                SOS_TOKEN, 
                device=delta.device
            )

##### --- Find Maximum Value and Index for each Point---
    valid_mask = G_neighbor_index_matrix != -1
    delta_prev = delta[:, ref_index]

    invalid_score = -torch.inf
    score_g_9 = torch.full((G, 9), invalid_score, dtype=torch.float32, device=delta.device)

    # 9 Directions
    for neighbor_pos in range(9):
        opposite_neighbor_pos = utils.opposite(neighbor_pos)
        neighbor_indices = G_neighbor_index_matrix[:, neighbor_pos]
        mask = valid_mask[:, neighbor_pos]

        # If there is any available neighbor
        if mask.any():
            gathered_deltas = delta_prev[neighbor_indices[mask]]

            if mode == 'MARKOV':
                score_g_9[mask, neighbor_pos] = gathered_deltas
                
            elif mode == 'TRANSFORMER':
                gather_transition = transition_log_prob_G_9[neighbor_indices[mask], opposite_neighbor_pos]
                score_g_9[mask, neighbor_pos] = gathered_deltas + gather_transition

    # Find Max and Argmax of score for each Reference Point
    max_value, G_winner_neighbor_relative_position = torch.max(score_g_9, dim=1)
    row_indices_j = torch.arange(G, device=G_neighbor_index_matrix.device)
    G_winner_neighbor_index = G_neighbor_index_matrix[
        row_indices_j,
        G_winner_neighbor_relative_position
    ]

    DEBUG = False
    if DEBUG:
        if t == 1:
            print("score_g_9: ", score_g_9)
            #print("G_winner_neighbor_relative_position: ", G_winner_neighbor_relative_position)
            #print("max_value: ", max_value)
    
    return G_winner_neighbor_index, max_value

def update_delta_and_path(
    t: int, 
    ref_index: int, 
    tgt_index: int, 
    delta: torch.Tensor, 
    path: torch.Tensor, 
    G_winner_neighbor_index: torch.Tensor, 
    max_value: torch.Tensor,
    current_emission_log_prob: torch.Tensor
):

    T = path.shape[1]

##### --- Update delta ---
    delta[:, tgt_index] = current_emission_log_prob + max_value

##### --- Update path
    path[:, T-1-t : T-1 : 1, tgt_index] = path[G_winner_neighbor_index, T-t : T : 1, ref_index]

    return delta, path