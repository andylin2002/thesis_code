from typing import List, Dict, Any, Optional
import torch
import numpy as np

from . import em_calculator as emc
from markov_model.uniform_markov import generate_uniform_markov_trajectory
from transformer.transformer_tool import generate_transformer_trajectory

TypeTrajectory = torch.Tensor
TypePropParams = Dict[str,torch.Tensor]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EM_Algorithm:
    def __init__(
            self, 
            feature_matrix: torch.Tensor, 
            config: Dict[str, Any], 
            reference_grid: torch.Tensor,
            context: Dict[str, Any], 
            model: Optional[torch.nn.Module], 
            mode: str, 
            directions_vectors: np.ndarray
        ):

        self.feature_matrix = feature_matrix
        self.config = config
        self.reference_grid = reference_grid 
        self.context = context
        self.model = model
        self.mode = mode
        self.directions_vectors = directions_vectors

        N_DIRECTIONS = self.directions_vectors.shape[0]
        self.SOS_TOKEN = N_DIRECTIONS

    ##### --- Get Q and T ---
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']

        self.ap_locations = emc.get_ap_locations(self.config, self.num_ap, DEVICE)

    ##### --- Initialization ---
        self.MEPLL_PropParams = -torch.inf
        self.MEPLL_Trajectory = -torch.inf
        self.MEPLL_record = -torch.inf

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

        if self.context['last_predicted_point'] == None: # Fisrt round
            power_q = self.feature_matrix[:, 0, 0]
            start_point = emc.select_initial_position(self.config, self.reference_grid, power_q, DEVICE)

        else: # Not the fisrt round
            start_point = self.context['last_predicted_point']

        if self.mode == 'MARKOV':
            trajectory = (
                generate_uniform_markov_trajectory(
                    config=self.config, 
                    reference_grid=self.reference_grid, 
                    start_point=start_point, 
                    device=DEVICE
                )
            )
        elif self.mode == 'TRANSFORMER':
            trajectory = (
                generate_transformer_trajectory(
                    model=self.model,
                    start_point=start_point,
                    directions_vectors=self.directions_vectors,
                    T_length=self.num_sample,
                    SOS_TOKEN=self.SOS_TOKEN,
                    device=DEVICE
                )
            )

            print("TRANSFORMER: ", trajectory)

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
    
    def _findPropParams_step(self):

        trajectory = self.trajectory
        propagation_params = self.propagation_params

        power_qt = self.feature_matrix[:, :, 0]
        angle_qt = self.feature_matrix[:, :, 1]
        delay_qt = self.feature_matrix[:, :, 2]
        
    ##### --- Initialize ---

        gamma_qtk = self.context['APs_LOS_ratio']
        L_qt = emc.calculate_L_tq(trajectory, self.ap_locations)

        angle_qt_mean = emc.calculate_angle_qt_mean(trajectory, self.ap_locations)
        angle_qt1_mean = angle_qt_mean.unsqueeze(2)

    ##### --- Maximize 'Marginal Emission Probability Log Likelihood' until converge ---
        MAX_MEPLL_PropParams = -torch.inf

        while True:
            propagation_params_old = {k: v.clone() for k, v in propagation_params.items()}
            gamma_qtk_old = gamma_qtk.clone()

            power_qk_average = emc.calculate_weighted_average(power_qt, gamma_qtk)
            L_qk_average =     emc.calculate_weighted_average(L_qt, gamma_qtk)

        ##### --- Parameters Update related to Power ---
            propagation_params['alpha_qk'] = (
                emc.calculate_alpha_qk(
                    power_qt, 
                    power_qk_average, 
                    L_qt, 
                    L_qk_average, 
                    gamma_qtk
                )
            )
            propagation_params['beta_qk'] = (
                emc.calculate_beta_qk(
                    propagation_params['alpha_qk'], 
                    power_qk_average, 
                    L_qk_average
                )
            )
            power_qtk_mean = (
                emc.calculate_power_qtk_mean(
                    propagation_params['alpha_qk'], 
                    propagation_params['beta_qk'], 
                    L_qt
                )
            )
            propagation_params['power_qk_var'] = (
                emc.calculate_power_qk_var(
                    power_qt,
                    power_qtk_mean,   
                    gamma_qtk
                )
            ) 

            power_q1k_var = propagation_params['power_qk_var'].unsqueeze(1)
            
            power_distribution_qtk = (
                emc.build_gaussian_distribution(
                    power_qtk_mean, 
                    power_q1k_var
                )
            )
            
        ##### --- Parameters Update related to Angle ---
            propagation_params['angle_k_var'] = (
                emc.calculate_angle_k_var(
                    angle_qt_mean, 
                    angle_qt, 
                    gamma_qtk
                )
            )
            angle_11k_var = propagation_params['angle_k_var'].view(1, 1, -1)

            angle_distribution_qtk = (
                emc.build_gaussian_distribution(
                    angle_qt1_mean, 
                    angle_11k_var
                )
            )

        ##### --- Parameters Update related to Delay ---
            propagation_params['delay_k_mean'] = (
                emc.calculate_delay_k_mean(
                    delay_qt, 
                    gamma_qtk
                )
            )
            propagation_params['delay_k_var'] = (
                emc.calculate_delay_k_var(
                    propagation_params['delay_k_mean'], 
                    delay_qt, 
                    gamma_qtk
                )
            )
            
            delay_11k_mean = propagation_params['delay_k_mean'].view(1, 1, -1)
            delay_11k_var = propagation_params['delay_k_var'].view(1, 1, -1)
            
            delay_distribution_qtk = (
                emc.build_gaussian_distribution(
                    delay_11k_mean, 
                    delay_11k_var
                )
            )
            
        ##### --- Parameters Update related to Global LOS ratio ---
            propagation_params['pi_global_LOS_ratio'] = emc.calculate_pi(gamma_qtk)

        ##### --- Parameters Update related to AP's LOS ratio ---
            gamma_qtk_new = (
                emc.calculate_gamma_qtk(
                    propagation_params['pi_global_LOS_ratio'], 
                    power_distribution_qtk, 
                    angle_distribution_qtk, 
                    delay_distribution_qtk, 
                    power_qt, 
                    angle_qt, 
                    delay_qt
                )
            )

        ##### --- Calculate 'Marginal Emission Probability Log Likelihood' for PropParams ---
            MEPLL_PropParams_new = (
                emc.calculate_MEPLL_PropParams(
                    propagation_params['pi_global_LOS_ratio'], 
                    power_distribution_qtk, 
                    angle_distribution_qtk, 
                    delay_distribution_qtk, 
                    power_qt, 
                    angle_qt, 
                    delay_qt, 
                    gamma_qtk_new
                )
            )

        ##### --- Check Whether 'findPropParams_step' is convergent ---
            if MEPLL_PropParams_new > MAX_MEPLL_PropParams + 1e-6:
                MAX_MEPLL_PropParams = MEPLL_PropParams_new
                gamma_qtk = gamma_qtk_new
                continue
            else:
                self.propagation_params = propagation_params_old 
                gamma_qtk = gamma_qtk_old
                break

        # END while

    ##### --- Update AP's LOS ratio ---
        self.context['APs_LOS_ratio'].copy_(gamma_qtk)

    ##### --- Found Local Limit Point of Parameters ---
        self.propagation_params = propagation_params
    ##### --- Record best MEPLL_PropParams ---
        self.MEPLL_PropParams = MAX_MEPLL_PropParams
        
    def _findTrajectory_step(self):

        feature_matrix = self.feature_matrix
        config = self.config
        reference_grid = self.reference_grid
        propagation_params = self.propagation_params
        ap_locations = self.ap_locations

        T = self.num_sample
        G = self.reference_grid.shape[0]

    ##### --- Fisrt: Construct G * T Emission Probability step ---

        emission_probability_gt = (
            emc.calculate_emission_probability(
                feature_matrix,
                reference_grid, 
                propagation_params,
                ap_locations, 
                DEVICE
            )
        )

    ##### --- Second: PingPong Updating step ---
        G_index = torch.arange(G).to(DEVICE)
        G_neighbor_index_matrix = emc.get_all_neighbor_indices(config, G_index, DEVICE) # shape: (G, 9)

        delta = torch.full((G, 2), -torch.inf, dtype=torch.float32, device=DEVICE)
        path = torch.full((G, T, 2), -1, dtype=torch.long, device=DEVICE)

        delta[:, 0] = emission_probability_gt[:, 0]

        G_index_stacked = torch.stack([G_index, G_index], dim=1)
        path[:, -1, :] = G_index_stacked

        for t in range(1, T):
            # Ping-Pong Structure
            ref_index = (t + 1) % 2
            tgt_index = t % 2

            current_emission_log_prob = emission_probability_gt[:, t]

            G_winner_neighbor_index, max_value = (
                emc.get_winner_neighbor_info(
                    t, 
                    reference_grid, 
                    ref_index, 
                    G_neighbor_index_matrix, 
                    delta, 
                    path, 
                    self.model,    
                    self.mode, 
                    self.SOS_TOKEN
                )
            )
            delta, path = (
                emc.update_delta_and_path(
                    t, 
                    ref_index, 
                    tgt_index, 
                    delta, 
                    path, 
                    G_winner_neighbor_index, 
                    max_value, 
                    current_emission_log_prob
                )
            )

            if t == (T - 1):
                MEPLL_Trajectory, chosen_index = torch.max(delta[:, tgt_index], dim=0)
                trajectory_index_sequence = path[chosen_index, :, tgt_index]
                self.trajectory = reference_grid[trajectory_index_sequence]

        self.MEPLL_Trajectory = MEPLL_Trajectory

        
    def _check_convergence(self):

        current_MEPLL = self.MEPLL_PropParams + self.MEPLL_Trajectory
    
        diff = abs(self.MEPLL_record - current_MEPLL)

        if diff < 1e-6:
            convergence = True
            
        else:
            convergence = False

        self.MEPLL_record = current_MEPLL
        
        return convergence
