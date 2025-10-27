from typing import List, Dict, Any, Optional
import torch

from . import em_calculator as emc
from markov_model.uniform_markov import generate_uniform_markov_trajectory

TypeTrajectory = torch.Tensor
TypePropParams = Dict[str,torch.Tensor]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# (TODO : 需要解決MEPLL震盪導致無法收斂的狀況)
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
        
        #trajectory = torch.zeros(self.num_sample, 2, dtype=torch.float32, device=DEVICE) # (TODO: add transformer)

        #"""
        # (FIXME)
        if self.context['current_round'] == 0: # Fisrt round
            power_q = self.feature_matrix[:, 0, 0]
            start_point = emc.select_initial_position(self.config, self.reference_grid, power_q, DEVICE)
            trajectory = generate_uniform_markov_trajectory(self.config, self.reference_grid, start_point, DEVICE)

        else: # Not the fisrt round
            start_point = self.context['last_predicted_point']
            trajectory = generate_uniform_markov_trajectory(self.config, self.reference_grid, start_point, DEVICE)
        
        #"""
        
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

        trajectory = self.trajectory
        propagation_params = self.propagation_params
        config = self.config

        power_qt = self.feature_matrix[:, :, 0]
        angle_qt = self.feature_matrix[:, :, 1]
        delay_qt = self.feature_matrix[:, :, 2]
        
        
    ##### --- Initialize ---
        T = self.num_sample
        Q = self.num_ap
        K = 2 # LOS and NLOS

        gamma_qtk = self.context['APs_LOS_ratio']
        L_qt = emc.calculate_L_tq(config, trajectory, Q)

        angle_qt_mean = emc.calculate_angle_qt_mean(config, trajectory, Q)
        angle_qtk_mean = angle_qt_mean.unsqueeze(2)

    ##### --- Maximize 'Marginal Emission Probability Log Likelihood' until converge ---
        MAX_MEPLL_PropParams = -torch.inf
        try_count = config['EM_M_STEP_TRY']

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
            if MEPLL_PropParams > MAX_MEPLL_PropParams:
                MAX_MEPLL_PropParams = MEPLL_PropParams
                try_count = config['EM_M_STEP_TRY']
                continue

            if try_count == 0:
                break

            try_count -= 1

    ##### --- Update AP's LOS ratio ---
        self.context['APs_LOS_ratio'].copy_(gamma_qtk)

    ##### --- Found Local Limit Point of Parameters ---
        self.propagation_params = propagation_params
        
    def _findTrajectory_step(self): # (TODO)

        trajectory = self.trajectory  
        config = self.config
        reference_grid = self.reference_grid

        T = self.num_sample
        Q = self.num_ap
        K = 2 # LOS and NLOS
        G = self.reference_grid.shape[0]

    ##### --- Fisrt: Construct G * T Emission Probability ---

        # (TODO: calculate_emission_probability in emc)
        emission_probability_gt = emc.calculate_emission_probability(self.feature_matrix, 
                                                                     self. config, 
                                                                     self.reference_grid, 
                                                                     self.propagation_params)

    ##### --- Second: Implement Vertebi Algorithm ---
        Candidate_Path: List[torch.Tensor] = [
            torch.empty(0, 2, dtype=torch.float32, device=DEVICE) 
            for _ in range(G)
        ]

        # delta會存t-1狀態（在[G, 0]）並用其算出t狀態（在[G, 1]），再將t狀態的結果存到t-1狀態（[G, 0] = [G, 1]）
        delta_g2 = torch.full((G, 2), -torch.inf, dtype=torch.float32, device=DEVICE)

        # batch = G，所有 G 同步計算（TODO: 需要克服稀疏狀況）
        for t in range(T):
            current_emission_log_prob = emission_probability_gt[:, t]

            # 算出第 t 步的 delta、紀錄哪一個點（i）造成最大 delta
            if t == 0: 
                delta_g2[:, 0] = current_emission_log_prob
                
            if t > 0:
                
                pass

            # delta推進一步
            delta_g2[:, 0] = delta_g2[:, 1]

            # 更新 Candidate_Path "concat(prev, i)"

            pass

            

        pass
        
    def _check_convergence(self): # (TODO)
        
        return False
