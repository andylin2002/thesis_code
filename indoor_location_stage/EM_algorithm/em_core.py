import numpy as np
from typing import Dict, Any, Optional
import torch

TypeTrajectory = torch.Tensor
TypePropParams = Dict[str,torch.Tensor]

# (TODO)
class EM_Algorithm:
    def __init__(
            self, 
            feature_matrix: torch.Tensor, 
            reference_grid: torch.Tensor,
            gamma_APs_LOS_ratio: torch.Tensor,
            config: Dict[str, Any]
        ):

        self.feature_matrix = feature_matrix
        self.reference_grid = reference_grid 
        self.gamma_APs_LOS_ratio = gamma_APs_LOS_ratio
        self.config = config

    ##### --- Get Q and T ---
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']

    ##### --- Initialization ---
        self.trajectory = self._initialize_Trajectory() # 馬可夫鏈初始化
        self.propagation_params = self._initialize_PropParams()

    def run_em_iterations(self) -> Optional[TypeTrajectory]:

        for i in range(self.config['EM_MAX_ITER']):
            self._findPropParams_step()
            self._findTrajectory_step()
            
            if self._check_convergence():
                break

        return self.trajectory

    
    def _initialize_Trajectory(self) -> Optional[TypeTrajectory]:
        
        trajectory = torch.zeros(self.num_sample, dtype=torch.float32) # (TODO: add transformer)
        # 第一輪EM：x1選功率最大的AP位置
        # 第二輪EM：x1選上一輪EM的最後一個點
        # 後面x2~xT都用Transformer直接推論

        return trajectory

    def _initialize_PropParams(self) -> Optional[TypePropParams]:

        Q = self.num_ap
        K = 2 # LOS and NLOS

        alpha_qk =              torch.zeros(Q, K, dtype=torch.float32)
        beta_qk =               torch.zeros(Q, K, dtype=torch.float32)
        power_qk_std_dev =      torch.zeros(Q, K, dtype=torch.float32)
        pi_global_LOS_ratio =   torch.zeros(K, dtype=torch.float32)
        angle_k_std_dev =       torch.zeros(K, dtype=torch.float32)
        delay_k_mean =          torch.zeros(K, dtype=torch.float32)
        delay_k_std_dev =       torch.zeros(K, dtype=torch.float32)

        propagation_params = {
            'alpha_qk':             alpha_qk,               # shape: (Q, K)
            'beta_qk':              beta_qk,                # shape: (Q, K)
            'power_qk_std_dev':     power_qk_std_dev,       # shape: (Q, K)
            'pi_global_LOS_ratio':  pi_global_LOS_ratio,    # shape: (K)
            'angle_k_std_dev':      angle_k_std_dev,        # shape: (K)              
            'delay_k_mean':         delay_k_mean,           # shape: (K)    
            'delay_k_std_dev':      delay_k_std_dev,        # shape: (K)        
        }

        return propagation_params
    
    def _findPropParams_step(self):

        pass
        
    def _findTrajectory_step(self):

        pass
        
    def _check_convergence(self):
        
        return False
    
    def _construct_emission_log_likelihood(self, mean, std_div) -> torch.Tensor:

        pass