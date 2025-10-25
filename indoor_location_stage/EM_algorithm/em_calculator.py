import torch
from typing import Dict, Any, Optional
from torch.distributions import Normal

TypeTrajectory = torch.Tensor
    
# (TODO)
# (TODO: 讓進到build_gaussian_distribution的mean, var都是qtk形狀)
def build_gaussian_distribution(mean: torch.Tensor, variance: torch.Tensor) -> Normal:
    """
    建構 PyTorch Normal 分佈物件。
    
    :param mean: 均值 (mu)。
    :param variance: 變異數 (sigma^2)。
    :return: Normal 分佈實例。
    """
    # 變異數必須先轉換成標準差 (std_dev = sqrt(variance))
    std_dev = torch.sqrt(variance)
    
    # 為了數值穩定性，確保標準差不是零
    std_dev = torch.clamp(std_dev, min=1e-6)
    
    # 建立分佈物件
    gaussian_dist = Normal(loc=mean, scale=std_dev)
    
    return gaussian_dist

def calculate_L_tq(self, trajectory: TypeTrajectory) -> torch.Tensor:

    ##### --- Prepare AP's position and trajectory ---
        ap_locations_list = []
        for ap_id in range(1, self.num_ap + 1):
            ap_key = f"AP_{ap_id}"
            if ap_key in self.config['ACCESS_POINTS']:
                ap_locations_list.append(self.config['ACCESS_POINTS'][ap_key]['LOCATION_M'])
            else:
                raise ValueError(f"Missing location data for AP ID {ap_id}")
            
        ap_locations = torch.tensor(ap_locations_list, dtype=torch.float32, device=DEVICE)
        ap_locations_expanded = ap_locations.unsqueeze(1)
        trajectory_expanded = trajectory.unsqueeze(0)

    ##### --- calculate the distance for each t and q ---
        squared_diff = (ap_locations_expanded - trajectory_expanded) ** 2
        distance_matrix = torch.sqrt(squared_diff.sum(dim=2))
        distance_matrix = torch.clamp(distance_matrix, min=1e-10)
        L_qt = torch.log10(distance_matrix)

        return L_qt

def calculate_weighted_mean(self, 
                            data_qt: torch.Tensor, 
                            gamma_qtk: torch.Tensor) -> torch.Tensor:
    """
    計算加權平均值 (例如 power_qk_mean)。
    :param data_qt: 特徵數據，形狀 (Q, T) 或 (Q, T, F)
    :param gamma_qtk: 後驗責任，形狀 (Q, T, K) 或 (Q, T)
    :return: 加權平均值，形狀 (Q, K) 或 (K)
    """
    pass

# ----------------------------------------------------
# --- 參數更新函式 (M-step) ---
# ----------------------------------------------------

# Power/RSS 相關參數

def calculate_alpha_qk(self, 
                    power_qt: torch.Tensor, 
                    power_qk_mean: torch.Tensor, 
                    L_qt: torch.Tensor, 
                    L_qk_mean: torch.Tensor, 
                    gamma_qtk: torch.Tensor) -> torch.Tensor:
    """
    計算路徑損耗指數 alpha_qk (對應論文公式 223)。
    """
    pass

def calculate_beta_qk(self, 
                    alpha_qk: torch.Tensor, 
                    power_qk_mean: torch.Tensor, 
                    L_qk_mean: torch.Tensor) -> torch.Tensor:
    """
    計算參考路徑損耗 beta_qk (對應論文公式 224)。
    """
    pass

def calculate_power_qk_std_dev(self, 
                            alpha_qk: torch.Tensor, 
                            beta_qk: torch.Tensor, 
                            power_qt: torch.Tensor, 
                            L_qt: torch.Tensor, 
                            gamma_qtk: torch.Tensor) -> torch.Tensor:
    """
    計算 RSS 方差 sigma_s,q,k^2 (對應論文公式 289)。
    """
    pass

# Angle 相關參數

def calculate_angle_k_std_dev(self, 
                            angle_qt: torch.Tensor, 
                            gamma_qtk: torch.Tensor) -> torch.Tensor:
    """
    計算 AoD 方差 sigma_theta,k^2 (對應論文公式 290)。
    """
    pass

# Delay 相關參數

def calculate_delay_k_mean(self, 
                            delay_qt: torch.Tensor, 
                            gamma_qtk: torch.Tensor) -> torch.Tensor:
    """
    計算延遲擴散均值 mu_k (對應論文公式 291)。
    """
    pass

def calculate_delay_k_std_dev(self, 
                            delay_k_mean: torch.Tensor, 
                            delay_qt: torch.Tensor, 
                            gamma_qtk: torch.Tensor) -> torch.Tensor:
    """
    計算延遲擴散協方差 Sigma_k (對應論文公式 292)。
    """
    pass