from typing import List, Dict, Any, Optional
import torch
import numpy as np
import torch.nn.functional as F

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
        self.ap_orientations = torch.tensor(
            [data.get('ORIENTATION_DEG', 0) for data in self.ap_data.values()], 
            dtype=torch.float32, 
            device=DEVICE
        )

    ##### --- Initialization ---
        self.MEPLL_PropParams = -torch.inf
        self.MEPLL_Trajectory = -torch.inf
        self.MEPLL_record = -torch.inf

        self.propagation_params = self._initialize_PropParams()
        self.trajectory = self._initialize_Trajectory()

        DEBUG = True
        if DEBUG:
            print("init self.trajectory: ", self.trajectory)

    def run_em_iterations(self) -> Optional[TypeTrajectory]:

        for i in range(self.config['EM_MAX_ITER']):
            self._findPropParams_step()
            self._findTrajectory_step()
            
            if self._check_convergence():
                break

        return self.trajectory
    
    def _initialize_PropParams(self) -> Optional[TypePropParams]:

        Q = self.num_ap
        T = self.num_sample
        K = 2 # LOS and NLOS
        MIN_VAR = 1

    ##### Initialize all learned propagation parameters
        alpha_qk =      torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        beta_qk =       torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        power_qk_var =  torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        angle_k_var =   torch.zeros(K, dtype=torch.float32, device=DEVICE)
        delay_k_mean =  torch.zeros(K, dtype=torch.float32, device=DEVICE)
        delay_k_var =   torch.zeros(K, dtype=torch.float32, device=DEVICE)
        pi_k =          torch.full((K, ), 0.5, dtype=torch.float32, device=DEVICE)
        gamma_qtk =     torch.full((Q, T, K), 0.5, dtype=torch.float32, device=DEVICE)

    ##### Extract features from the input matrix
        power_qt = self.feature_matrix[:, :, 0]
        angle_qt = self.feature_matrix[:, :, 1]
        delay_qt = self.feature_matrix[:, :, 2]

    ##### Initialize "POWER" parameters
        alpha_qk, beta_qk = (
            emc.calculate_init_alpha_and_beta_qk(
                self.reference_grid, 
                self.ap_locations, 
                power_qt
            )
        )
        power_qk_var = (
            emc.calculate_init_power_qk_var(
                self.reference_grid, 
                self.ap_locations, 
                alpha_qk, 
                beta_qk, 
                power_qt
            )
        )

    ##### Initialize "ANGLE" parameters
        angle_k_var = (
            emc.calculate_init_angle_k_var(
                self.reference_grid, 
                self.ap_locations, 
                self.ap_orientations,
                angle_qt
            )
        )

    ##### Initialize "DELAY" parameters
        # Cluster Delay data globally and assign means and variances
        delay_flat = delay_qt.flatten()
        delay_means, delay_vars = emc.estimate_two_gaussians(delay_flat)
        delay_k_mean.copy_(delay_means.to(DEVICE))
        delay_k_var.copy_(delay_vars.clamp(min=MIN_VAR))

    ##### Initialize "gamma_qtk"
        delay_11k_mean = delay_k_mean.view(1, 1, -1)
        delay_11k_var = delay_k_var.view(1, 1, -1)
        delay_distribution_qtk = (
            emc.build_gaussian_distribution(
                delay_11k_mean, 
                delay_11k_var
            )
        )
        
        gamma_qtk = (
            emc.calculate_init_gamma_qtk(
                delay_distribution_qtk, 
                delay_qt
            )
        )
        
    ##### Structure and return the propagation parameters dictionary
        propagation_params = {
            'alpha_qk':             alpha_qk,               # shape: (Q, K)
            'beta_qk':              beta_qk,                # shape: (Q, K)
            'power_qk_var':         power_qk_var,           # shape: (Q, K)
            'angle_k_var':          angle_k_var,            # shape: (K)              
            'delay_k_mean':         delay_k_mean,           # shape: (K)    
            'delay_k_var':          delay_k_var,            # shape: (K)   
            'pi_k':                 pi_k,                   # shape: (K)
            'gamma_qtk':            gamma_qtk,              # shape: (Q, T, K)
        }

        DEBUG = True
        if DEBUG:
            print("=== Initial Params ===")
            print("alpha_qk: ", propagation_params['alpha_qk']) # DEBUG
            print("beta_qk: ", propagation_params['beta_qk']) # DEBUG
            print("power_qk_var: ", propagation_params['power_qk_var']) # DEBUG
            print("angle_k_var: ", propagation_params['angle_k_var']) # DEBUG

        return propagation_params
    
    def _initialize_Trajectory(self) -> Optional[TypeTrajectory]:

        T = self.num_sample

        self.trajectory = torch.zeros(T, 2, dtype=torch.float32, device=DEVICE)
        self._findTrajectory_step()

        return self.trajectory

    def _findPropParams_step(self):
        DEBUG = True
        if DEBUG:
            print("!!!Prop_Stage!!!")

        trajectory = self.trajectory
        propagation_params = self.propagation_params

        power_qt = self.feature_matrix[:, :, 0]
        angle_qt = self.feature_matrix[:, :, 1]
        delay_qt = self.feature_matrix[:, :, 2]
        
    ##### --- Initialize Constants ---
        L_qt = emc.calculate_L_qt(trajectory, self.ap_locations)

        angle_qt_mean = emc.calculate_angle_qt_mean(trajectory, self.ap_locations, self.ap_orientations)
        angle_qt1_mean = angle_qt_mean.unsqueeze(2)

    ##### --- Maximize 'Marginal Emission Probability Log Likelihood' until converge ---
        MAX_MEPLL_PropParams = -torch.inf

        while True:
            propagation_params_old = {k: v.clone() for k, v in propagation_params.items()}

            power_qk_average = (
                emc.calculate_weighted_average(
                    power_qt, 
                    propagation_params['gamma_qtk']
                )
            )
            L_qk_average = (
                emc.calculate_weighted_average(
                    L_qt, 
                    propagation_params['gamma_qtk']
                )
            )

        ##### --- Parameters Update related to Power ---
            propagation_params['alpha_qk'] = (
                emc.calculate_alpha_qk(
                    power_qt, 
                    power_qk_average, 
                    L_qt, 
                    L_qk_average, 
                    propagation_params['gamma_qtk']
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
                    propagation_params['gamma_qtk']
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
                    propagation_params['gamma_qtk']
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
                    propagation_params['gamma_qtk']
                )
            )
            propagation_params['delay_k_var'] = (
                emc.calculate_delay_k_var(
                    propagation_params['delay_k_mean'], 
                    delay_qt, 
                    propagation_params['gamma_qtk']
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
            propagation_params['pi_k'] = emc.calculate_pi(propagation_params['gamma_qtk'])

            DEBUG = False
            if DEBUG:
                q = 0
                k = 0
                t = 0
                print("gamma_qtk: ", propagation_params['gamma_qtk'][:, 1:10, :]) # DEBUG

                # print("alpha_qk: ", propagation_params['alpha_qk'][q]) # DEBUG
                # print("beta_qk: ", propagation_params['beta_qk'][q]) # DEBUG
                # print("power_qt: ", power_qt[q][t])
                # print("power_qtk_mean: ", power_qtk_mean[q][t])
                # print("power_qk_var: ", propagation_params['power_qk_var'][q]) # DEBUG

                # print("angle_k_var: ", propagation_params['angle_k_var']) # DEBUG

                # print("delay_k_mean: ", propagation_params['delay_k_mean'])
                # print("delay_k_var", propagation_params['delay_k_var'])

        ##### --- Calculate 'Marginal Emission Probability Log Likelihood' for PropParams ---
            MEPLL_PropParams_new = (
                emc.calculate_MEPLL_PropParams(
                    propagation_params['pi_k'], 
                    power_distribution_qtk, 
                    angle_distribution_qtk, 
                    delay_distribution_qtk, 
                    power_qt, 
                    angle_qt, 
                    delay_qt,
                )
            )

        ##### --- Parameters Update related to AP's LOS ratio ---
            propagation_params['gamma_qtk'] = (
                emc.calculate_gamma_qtk(
                    propagation_params['pi_k'], 
                    power_distribution_qtk, 
                    angle_distribution_qtk, 
                    delay_distribution_qtk, 
                    power_qt, 
                    angle_qt, 
                    delay_qt
                )
            )

            DEBUG = True
            if DEBUG:
                print("===== Iteration =====\n")
                print(f"  MEPLL_New: {MEPLL_PropParams_new:.4f} vs Old_Max: {MAX_MEPLL_PropParams:.4f}")
                #print(f"  Alpha (LOS/NLOS):\n{propagation_params['alpha_qk'][3]}")
                #print(f"  Beta (LOS/NLOS):\n{propagation_params['beta_qk'][3]}")
                #print(f"  Power Var (LOS/NLOS):\n{propagation_params['power_qk_var'][3]}")
                #print(f"  Delay Var (LOS/NLOS):\n{propagation_params['delay_k_var']}")
                #print(f"  Global Pi (LOS/NLOS): {propagation_params['pi_k']}")
                #print(f"  Gamma (LOS/NLOS): {propagation_params['gamma_qtk'][0]}")
            
        ##### --- Check Whether 'findPropParams_step' is convergent ---
            if MEPLL_PropParams_new > MAX_MEPLL_PropParams + 1e-4:
                MAX_MEPLL_PropParams = MEPLL_PropParams_new
                continue
            else:
                self.propagation_params = propagation_params_old 
                #propagation_params['gamma_qtk'] = propagation_params_old['gamma_qtk']
                break

        # END while

    ##### --- Found Local Limit Point of Parameters ---
        self.propagation_params = propagation_params
    ##### --- Record best MEPLL_PropParams ---
        self.MEPLL_PropParams = MAX_MEPLL_PropParams
        
    def _findTrajectory_step(self):
        DEBUG = True
        if DEBUG:
            print("!!!Traj_Stage!!!")

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
                self.ap_orientations, 
                DEVICE
            )
        )

        DEBUG = False
        if DEBUG:
            print("emission_probability_gt[:, 0]", emission_probability_gt[:, 0])

    ##### --- Second: PingPong Updating step ---
        # Construct Neighbor Table
        G_index = torch.arange(G).to(DEVICE)
        G_neighbor_index_matrix = emc.get_all_neighbor_indices(config, G_index, DEVICE) # shape: (G, 9)

        # Initialize delta and path
        delta = torch.full((G, 2), -torch.inf, dtype=torch.float32, device=DEVICE)
        path = torch.full((G, T, 2), -1, dtype=torch.long, device=DEVICE)

        delta[:, 0] = emission_probability_gt[:, 0]

        G_index_stacked = torch.stack([G_index, G_index], dim=1)
        path[:, -1, :] = G_index_stacked

        # delta and path are Update over 't-1' Iteration
        for t in range(1, T):
            # Ping-Pong Structure
            ref_index = (t + 1) % 2
            tgt_index = t % 2

            current_emission_log_prob = emission_probability_gt[:, t]

            # Find the Max and Argmax of 'delta + logP' for each Reference Point
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

            # Update
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

            DEBUG = False
            if DEBUG:
                print("path of idx70: ", path[70, :, tgt_index])

            DEBUG = False
            if DEBUG:
                if t == 1:
                    # 獲取當前時間步 t 的累積對數機率 (Delta)
                    current_delta = delta[:, tgt_index] 
                    print("current_delta: ", current_delta)
                    
                    # 找出目前所有網格點中的最大值
                    max_log_prob = torch.max(current_delta).item()
                    
                    # 找出最大值的索引（即目前 Viterbi 認為最佳的位置）
                    best_grid_index = torch.argmax(current_delta).item()
                    best_coord = self.reference_grid[best_grid_index].cpu().numpy()
                    
                    print(f"\n--- Viterbi Step t={t} Analysis ---")
                    print(f"Max Log-Prob: {max_log_prob:.4f}")
                    print(f"Best Point Index (g): {best_grid_index}")
                    print(f"Best Coordinate (x, y): {best_coord}")
                    
                    # 輸出部分 Delta 矩陣，以便觀察各點的分佈情況
                    # 由於 Delta 矩陣 (G, 2) 很大，我們只輸出排名前 N 的點，或輸出部分子集。
                    
                    # 找出前 5 名的最佳點
                    top_N = 5
                    top_values, top_indices = torch.topk(current_delta, top_N)
                    
                    print(f"Top {top_N} Candidates (Index | Log-Prob | Coord):")
                    for i in range(top_N):
                        idx = top_indices[i].item()
                        coord = self.reference_grid[idx].cpu().numpy()
                        log_prob = top_values[i].item()
                        print(f"  {i+1}. Index {idx}: {log_prob:.4f} at {coord}")

                    # 輸出固定感興趣點的 Delta 值 (例如您的起始點和預測點)
                    
                    # 假設 grid_index_pred 和 grid_index_true 已在之前計算好
                    # print(f"  > PRED Index {grid_index_pred} Log-Prob: {current_delta[grid_index_pred].item():.4f}")
                    # print(f"  > TRUE Index {grid_index_true} Log-Prob: {current_delta[grid_index_true].item():.4f}")
                    
                    print("---------------------------------")

            # Find the Trajectory by the Largest delta
            if t == (T - 1):
                MEPLL_Trajectory, chosen_index = torch.max(delta[:, tgt_index], dim=0)
                trajectory_index_sequence = path[chosen_index, :, tgt_index]
                self.trajectory = reference_grid[trajectory_index_sequence]

        self.MEPLL_Trajectory = MEPLL_Trajectory

        DEBUG = False
        if DEBUG:
            print(self.trajectory)

        
    def _check_convergence(self):

        current_MEPLL = self.MEPLL_PropParams + self.MEPLL_Trajectory
        diff = abs(self.MEPLL_record - current_MEPLL)

        if diff < 1e-6:
            convergence = True
            
        else:
            convergence = False

        self.MEPLL_record = current_MEPLL
        
        return convergence
