import torch
from typing import Dict, Any, Optional
from torch.distributions import Normal

TypeTrajectory = torch.Tensor

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

    return reference_grid[start_grid_idx]

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
    
    return power_qk_var + 1e-6

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

#******************************************************************************#
#******************************************************************************#
#********************************************* --- findTrajectory Step --- ****#
#******************************************************************************#
#******************************************************************************#

import torch
from typing import Dict, Any

def get_all_neighbor_indices(
    config: Dict[str, Any],
    grid_indices_G: torch.Tensor, # <--- 接受形狀為 [G] 的張量
    device: torch.device
) -> torch.Tensor:
    """
    向量化版本：輸入 G 個網格索引 (形狀 [G])，輸出 G 個點各自的 9 個鄰近點索引 (形狀 [G, 9])。
    
    無效的鄰近點會用 -1 填充。
    """
    
    # 1. 從配置中獲取網格尺寸 W 和 H
    W = config['X_WIDTH']  # 網格寬度 (列數)
    H = config['Y_WIDTH']  # 網格高度 (行數)
    G = grid_indices_G.shape[0] # 輸入點的數量
    
    # 2. 將一維索引 [G] 轉換為二維座標 [G, 1]
    # grid_indices_G 必須是 long 類型
    grid_indices_G = grid_indices_G.long() 
    
    # [G] -> [G, 1]
    current_rows_G = (grid_indices_G // W).unsqueeze(1) 
    current_cols_G = (grid_indices_G % W).unsqueeze(1)
    
    # 3. 預先計算偏移量張量
    # 偏移量矩陣 (3x3 = 9 種組合)，形狀 [1, 9]
    delta_row_1_9 = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.long, device=device).unsqueeze(0)
    delta_col_1_9 = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.long, device=device).unsqueeze(0)
    
    # 4. 廣播運算計算所有 G * 9 個候選點的 (row, col)
    # [G, 1] + [1, 9] -> [G, 9]
    
    # 候選行座標 [G, 9]
    candidate_rows_G_9 = current_rows_G + delta_row_1_9
    # 候選列座標 [G, 9]
    candidate_cols_G_9 = current_cols_G + delta_col_1_9
    
    # 5. 邊界檢查 (使用掩碼)，形狀 [G, 9]
    mask_row_valid = (candidate_rows_G_9 >= 0) & (candidate_rows_G_9 < H)
    mask_col_valid = (candidate_cols_G_9 >= 0) & (candidate_cols_G_9 < W)
    valid_mask_G_9 = mask_row_valid & mask_col_valid
    
    # 6. 將有效的 (row, col) 轉換回一維索引 [G, 9]
    
    # 計算所有 9 個候選點的一維索引 (包含邊界外的值)
    # W 是 Python int，可以直接用於乘法
    all_candidate_indices_G_9 = candidate_rows_G_9 * W + candidate_cols_G_9
    
    # 7. 應用掩碼：無效的點設定為 -1
    # 創建一個填充了 -1 的張量作為基礎
    neighbor_indices = torch.full((G, 9), -1, dtype=torch.long, device=device)
    
    # 將所有有效索引放入結果張量中
    # 這裡我們使用 where 或直接的索引替換
    neighbor_indices[valid_mask_G_9] = all_candidate_indices_G_9[valid_mask_G_9]
    
    return neighbor_indices

def calculate_emission_probability(
        feature_matrix: torch.Tensor, 
        config: Dict[str, Any], 
        reference_grid: torch.Tensor,
        propagation_params: Dict[str, Any]
) -> torch.Tensor:

    pass

def calcultate_delta_info(
    t: int, 
    ref_index: int, 
    tgt_index: int, 
    G_neighbor_index_matrix: torch.Tensor, 
    delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    
    pass

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

    pass