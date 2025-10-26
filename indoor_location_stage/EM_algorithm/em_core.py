from typing import Dict, Any, Optional
import torch

from . import em_calculator as emc

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
        power_qk_var =          torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        pi_global_LOS_ratio =   torch.zeros(K, dtype=torch.float32, device=DEVICE)
        angle_k_var =           torch.zeros(K, dtype=torch.float32, device=DEVICE)
        delay_k_mean =          torch.zeros(K, dtype=torch.float32, device=DEVICE)
        delay_k_var =           torch.zeros(K, dtype=torch.float32, device=DEVICE)

        propagation_params = {
            'alpha_qk':             alpha_qk,               # shape: (Q, K)
            'beta_qk':              beta_qk,                # shape: (Q, K)
            'power_qk_var':         power_qk_var,           # shape: (Q, K)
            'pi_global_LOS_ratio':  pi_global_LOS_ratio,    # shape: (K)
            'angle_k_var':          angle_k_var,            # shape: (K)              
            'delay_k_mean':         delay_k_mean,           # shape: (K)    
            'delay_k_var':          delay_k_var,            # shape: (K)        
        }

        return propagation_params
    
    def _findPropParams_step(self): # (TODO)
        
        T = self.num_sample
        Q = self.num_ap
        K = 2 # LOS and NLOS

        trajectory = self.trajectory
        propagation_params = self.propagation_params
        config = self.config

        power_qt = self.feature_matrix[:, :, 0]
        angle_qt = self.feature_matrix[:, :, 1]
        delay_qt = self.feature_matrix[:, :, 2]
        
        
    ##### --- Initialize ---
        gamma_qtk = self.context['APs_LOS_ratio']
        L_qt = emc.calculate_L_tq(config, trajectory, Q)

        angle_qt_mean = emc.calculate_angle_qt_mean(config, trajectory, Q)
        angle_qtk_mean = angle_qt_mean.unsqueeze(2)

    ##### --- Maximize 'Marginal Emission Probability Log Likelihood' until converge ---
        MAX_MEPLL_PropParams = -torch.inf

        while True:
            power_qk_average = emc.calculate_weighted_average(power_qt, gamma_qtk)
            L_qk_average =     emc.calculate_weighted_average(L_qt, gamma_qtk)

        ##### --- Parameters Update related to Power ---
            propagation_params['alpha_qk'] = emc.calculate_alpha_qk(power_qt, power_qk_average, L_qt, L_qk_average, gamma_qtk)
            propagation_params['beta_qk'] = emc.calculate_beta_qk(propagation_params['alpha_qk'], power_qk_average, L_qk_average)

            power_qtk_mean = emc.calculate_power_qtk_mean(propagation_params['alpha_qk'], 
                                                                    propagation_params['beta_qk'], 
                                                                    L_qt)

            propagation_params['power_qk_var'] = emc.calculate_power_qk_var(power_qt,
                                                                            power_qtk_mean,   
                                                                            gamma_qtk) 

            power_qtk_var = propagation_params['power_qk_var'].unsqueeze(1)
            
            power_distribution_qtk = emc.build_gaussian_distribution(power_qtk_mean, power_qtk_var)
            
        ##### --- Parameters Update related to Angle ---
            propagation_params['angle_k_var'] = emc.calculate_angle_k_var(angle_qt_mean, angle_qt, gamma_qtk)
            angle_qtk_var = propagation_params['angle_k_var'].view(1, 1, -1)

            angle_distribution_qtk = emc.build_gaussian_distribution(angle_qtk_mean, angle_qtk_var)

        ##### --- Parameters Update related to Delay ---
            propagation_params['delay_k_mean'] = emc.calculate_delay_k_mean(delay_qt, gamma_qtk)
            propagation_params['delay_k_var'] = emc.calculate_delay_k_var(propagation_params['delay_k_mean'], 
                                                                          delay_qt, 
                                                                          gamma_qtk)
            
            delay_qtk_mean = propagation_params['delay_k_mean'].view(1, 1, -1)
            delay_qtk_var = propagation_params['delay_k_var'].view(1, 1, -1)
            
            delay_distribution_qtk = emc.build_gaussian_distribution(delay_qtk_mean, delay_qtk_var)
            
        ##### --- Parameters Update related to Global LOS ratio ---
            propagation_params['pi_global_LOS_ratio'] = emc.calculate_pi(gamma_qtk)

        ##### --- Parameters Update related to AP's LOS ratio ---
            gamma_qtk = emc.calculate_gamma_qtk(propagation_params['pi_global_LOS_ratio'], 
                                                power_distribution_qtk, 
                                                angle_distribution_qtk, 
                                                delay_distribution_qtk, 
                                                power_qt, 
                                                angle_qt, 
                                                delay_qt)

        ##### --- Calculate 'Marginal Emission Probability Log Likelihood' for PropParams ---
            MEPLL_PropParams = emc.calculate_MEPLL_PropParams(propagation_params['pi_global_LOS_ratio'], 
                                                              power_distribution_qtk, 
                                                              angle_distribution_qtk, 
                                                              delay_distribution_qtk, 
                                                              power_qt, 
                                                              angle_qt, 
                                                              delay_qt, 
                                                              gamma_qtk)

        ##### --- Check Whether 'findPropParams_step' is convergent ---
            MEPLL_difference = MEPLL_PropParams - MAX_MEPLL_PropParams

            if MEPLL_difference > 0:
                MAX_MEPLL_PropParams = MEPLL_PropParams

            if torch.abs(MEPLL_difference) < config['EM_M_STEP_TH']:
                break

    ##### --- Update AP's LOS ratio ---
        self.context['APs_LOS_ratio'].copy_(gamma_qtk)
        
    def _findTrajectory_step(self): # (TODO)

        pass
        
    def _check_convergence(self): # (TODO)
        
        return False
