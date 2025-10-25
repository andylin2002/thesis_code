from typing import Dict, Any, Optional
import torch

import em_calculator as emc

TypeTrajectory = torch.Tensor
TypePropParams = Dict[str,torch.Tensor]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# (TODO)
class EM_Algorithm:
    def __init__(
            self, 
            feature_matrix: torch.Tensor, 
            config: Dict[str, Any], 
            reference_grid: torch.Tensor,
            context: Dict[str, Any]
        ):

        self.feature_matrix = feature_matrix
        self.config = config
        self.reference_grid = reference_grid 
        self.context = context

    ##### --- Get Q and T ---
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']

    ##### --- Initialization ---
        self.trajectory = self._initialize_Trajectory()
        self.propagation_params = self._initialize_PropParams()

    def run_em_iterations(self) -> Optional[TypeTrajectory]:

        for i in range(self.config['EM_MAX_ITER']):
            self._findPropParams_step()
            self._findTrajectory_step()
            
            if self._check_convergence():
                break

        return self.trajectory

    
    def _initialize_Trajectory(self) -> Optional[TypeTrajectory]:
        
        trajectory = torch.zeros(self.num_sample, 2, dtype=torch.float32, device=DEVICE) # (TODO: add transformer)
        # 第一輪EM：x1選功率最大的AP位置
        # 第二輪EM：x1選上一輪EM的最後一個點
        # 後面x2~xT都用Transformer直接推論

        return trajectory

    def _initialize_PropParams(self) -> Optional[TypePropParams]:

        Q = self.num_ap
        K = 2 # LOS and NLOS

        alpha_qk =              torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        beta_qk =               torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        power_qk_std_dev =      torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        pi_global_LOS_ratio =   torch.zeros(K, dtype=torch.float32, device=DEVICE)
        angle_k_std_dev =       torch.zeros(K, dtype=torch.float32, device=DEVICE)
        delay_k_mean =          torch.zeros(K, dtype=torch.float32, device=DEVICE)
        delay_k_std_dev =       torch.zeros(K, dtype=torch.float32, device=DEVICE)

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
    
    def _findPropParams_step(self): # (TODO)
        
        T = self.num_sample
        Q = self.num_ap
        K = 2 # LOS and NLOS

        trajectory = self.trajectory
        propagation_params = self.propagation_params

        power_qt = self.feature_matrix[:, :, 0]
        angle_qt = self.feature_matrix[:, :, 1]
        delay_qt = self.feature_matrix[:, :, 2]
        
        
    ##### --- Initialize ---
        gamma_qtk = self.context['APs_LOS_ratio']
        L_qt = emc.calculate_L_tq(trajectory)

        while True:
            power_qk_mean = emc.calculate_weighted_mean(power_qt, gamma_qtk)
            L_qk_mean =     emc.calculate_weighted_mean(L_qt, gamma_qtk)

        ##### --- Parameters Update related to Power ---
            propagation_params['alpha_qk'] = emc.calculate_alpha_qk(power_qt, power_qk_mean, L_qt, L_qk_mean, gamma_qtk)
            propagation_params['beta_qk'] = emc.calculate_beta_qk(propagation_params['alpha_qk'], power_qk_mean, L_qk_mean)
            #記得要unsqueeze以下
            propagation_params['power_qk_std_dev'] = emc.calculate_power_qk_std_dev(propagation_params['alpha_qk'], 
                                                                                 propagation_params['beta_qk'], 
                                                                                 power_qt, 
                                                                                 L_qt, 
                                                                                 gamma_qtk)
            
            power_qtk_mean = emc.calculate_power_qtk_mean(propagation_params['alpha_qk'], 
                                                          propagation_params['beta_qk'], 
                                                          L_qt)
            power_distribution_qtk = emc.build_gaussian_distribution(power_qtk_mean, propagation_params['power_qk_std_dev'])
            
        ##### --- Parameters Update related to Angle ---
            #記得要unsqueeze以下
            propagation_params['angle_k_std_dev'] = emc.calculate_angle_k_std_dev(angle_qt, gamma_qtk)

            angle_qtk_mean = torch.zeros(Q, T, K, dtype=torch.float32, device=DEVICE)
            angle_distribution_qtk = emc.build_gaussian_distribution(angle_qtk_mean, propagation_params['angle_k_std_dev'])

        ##### --- Parameters Update related to Delay ---
            #記得要unsqueeze以下
            propagation_params['delay_k_mean'] = emc.calculate_delay_k_mean(delay_qt, gamma_qtk)
            propagation_params['delay_k_std_dev'] = emc.calculate_delay_k_std_dev(propagation_params['delay_k_mean'], 
                                                                                delay_qt, 
                                                                                gamma_qtk)
            
            delay_distribution_qtk = emc.build_gaussian_distribution(propagation_params['delay_k_mean'], propagation_params['delay_k_std_dev'])
            
        ##### --- Parameters Update related to Global LOS ratio ---
            propagation_params['pi_global_LOS_ratio'] = emc.calculate_pi(gamma_qtk)
            gamma_qtk = emc.calculate_gamma_qtk(propagation_params['pi_global_LOS_ratio'], 
                                                propagation_params['alpha_qk'], 
                                                propagation_params['beta_qk'], 
                                                propagation_params['power_qk_std_dev'], 
                                                propagation_params['angle_k_std_dev'], 
                                                propagation_params['delay_k_mean'], 
                                                propagation_params['delay_k_std_dev'], 
                                                power_qt, 
                                                angle_qt, 
                                                delay_qt)

        ##### --- Calculate 'Marginal Emission Probability Log Likelihood' for PropParams ---
            MEPLL_PropParams = emc.calculate_MEPLL_PropParams(propagation_params['pi_global_LOS_ratio'], 
                                                              )

            break




        # 最後更新會用到以下的東西
        """
        gamma_tqk = self.context['APs_LOS_ratio'] 
        new_gamma_tensor = torch.zeros(T, Q, K, dtype=torch.float32) # EM 計算結果
        gamma_tqk.copy_(new_gamma_tensor) #已確定可以直接改
        """

        pass
        
    def _findTrajectory_step(self): # (TODO)

        pass
        
    def _check_convergence(self): # (TODO)
        
        return False
    
    def _construct_emission_log_likelihood(self, mean, std_div) -> torch.Tensor: # (TODO)

        pass