from typing import List, Dict, Any, Optional
import torch
import numpy as np
import torch.nn.functional as F

from . import hard_em_utils
from .._common import grid_tools
from .._common import math_tools

from .._common.path_manager import ViterbiPathManager

TypeTrajectory = torch.Tensor
TypePropParams = Dict[str,torch.Tensor]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HardEM_Algorithm:
    def __init__(
            self, 
            feature_matrix: torch.Tensor, 
            config: Dict[str, Any], 
            reference_grid: torch.Tensor,
            model: Optional[torch.nn.Module], 
            mode: str, 
            directions_vectors: np.ndarray
        ):

        self.feature_matrix = feature_matrix
        self.config = config
        self.reference_grid = reference_grid
        self.model = model
        self.mode = mode
        self.directions_vectors = directions_vectors

        N_DIRECTIONS = self.directions_vectors.shape[0]
        self.SOS_TOKEN = N_DIRECTIONS

    ##### --- Get Q and T ---
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']

        self.ap_locations = grid_tools.get_ap_locations(self.config, self.num_ap, DEVICE)
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

        DEBUG = False
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
        """
        if self.config['SYSTEM_MODE'] == 'BASELINE':
        ##### --- Random Parameters ---
            alpha_qk = torch.rand((self.num_ap, 2), dtype=torch.float32, device=DEVICE) * 5.0 + 1.0
            beta_qk = torch.rand((self.num_ap, 2), dtype=torch.float32, device=DEVICE) * 60.0 - 90.0
            power_qk_var = torch.rand((self.num_ap, 2), dtype=torch.float32, device=DEVICE) * 14.0 + 1.0

            angle_k_var = torch.rand((2,), dtype=torch.float32, device=DEVICE) * 2950.0 + 50.0

            raw_delay_means = torch.rand((2,), dtype=torch.float32, device=DEVICE) * 90.0 + 10.0
            delay_k_mean, _ = torch.sort(raw_delay_means)
            delay_k_var = torch.rand((2,), dtype=torch.float32, device=DEVICE) * 49.0 + 1.0

            pi_k = torch.nn.functional.softmax(torch.randn((2,), dtype=torch.float32, device=DEVICE), dim=0)
            raw_gamma = torch.randn((self.num_ap, self.num_sample, 2), dtype=torch.float32, device=DEVICE)
            gamma_qtk = torch.nn.functional.softmax(raw_gamma, dim=-1)
        
        if self.config['SYSTEM_MODE'] == 'BASELINE':
        """
    ##### Initialize all learned propagation parameters
        alpha_qk =      torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        beta_qk =       torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        power_qk_var =  torch.zeros(Q, K, dtype=torch.float32, device=DEVICE)
        angle_k_var =   torch.zeros(K, dtype=torch.float32, device=DEVICE)
        delay_k_mean =  torch.zeros(K, dtype=torch.float32, device=DEVICE)
        delay_k_var =   torch.zeros(K, dtype=torch.float32, device=DEVICE)
        pi_k =          torch.full((K, ), 0.5, dtype=torch.float32, device=DEVICE)
        gamma_qtk =     torch.full((Q, T, K), 0.5, dtype=torch.float32, device=DEVICE)

    ##### --- Extract features from the input matrix ---
        power_qt = self.feature_matrix[:, :, 0]
        angle_qt = self.feature_matrix[:, :, 1]
        delay_qt = self.feature_matrix[:, :, 2]

    ##### --- Initialize "POWER" parameters ---
        alpha_qk, beta_qk = (
            hard_em_utils.calculate_init_alpha_and_beta_qk(
                self.reference_grid, 
                self.ap_locations, 
                power_qt
            )
        )
        power_qk_var = (
            hard_em_utils.calculate_init_power_qk_var(
                self.reference_grid, 
                self.ap_locations, 
                alpha_qk, 
                beta_qk, 
                power_qt
            )
        )

    ##### --- Initialize "ANGLE" parameters ---
        angle_k_var = (
            hard_em_utils.calculate_init_angle_k_var(
                self.reference_grid, 
                self.ap_locations, 
                self.ap_orientations,
                angle_qt
            )
        )

    ##### --- Initialize "DELAY" parameters ---
        # Cluster Delay data globally and assign means and variances
        delay_flat = delay_qt.flatten()
        delay_means, delay_vars = hard_em_utils.estimate_two_gaussians(delay_flat)
        delay_k_mean.copy_(delay_means.to(DEVICE))
        delay_k_var.copy_(delay_vars.clamp(min=MIN_VAR))

    ##### --- Initialize "gamma_qtk" ---
        delay_11k_mean = delay_k_mean.view(1, 1, -1)
        delay_11k_var = delay_k_var.view(1, 1, -1)

        mean_distance = torch.abs(delay_k_mean[0] - delay_k_mean[1])

        delay_11k_var = delay_11k_var * 1 * mean_distance.clamp(min=1.0)
        delay_distribution_qtk = (
            hard_em_utils.build_gaussian_distribution(
                delay_11k_mean, 
                delay_11k_var
            )
        )
        
        gamma_qtk = (
            hard_em_utils.calculate_init_gamma_qtk(
                delay_distribution_qtk, 
                delay_qt
            )
        )
        
    ##### --- Structure and return the propagation parameters dictionary ---
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

        DEBUG = False
        if DEBUG:
            print("=== Initial Params ===")
            print("alpha_qk: ", propagation_params['alpha_qk']) # DEBUG
            print("beta_qk: ", propagation_params['beta_qk']) # DEBUG
            print("power_qk_var: ", propagation_params['power_qk_var']) # DEBUG
            print("angle_k_var: ", propagation_params['angle_k_var']) # DEBUG
            print("delay_k_mean: ", propagation_params['delay_k_mean']) # DEBUG
            print("delay_k_var: ", propagation_params['delay_k_var']) # DEBUG
            #print("gamma_qtk[3]: ", propagation_params['gamma_qtk'][3]) # DEBUG

        return propagation_params
    
    def _initialize_Trajectory(self) -> Optional[TypeTrajectory]:

        T = self.num_sample

        self.trajectory = torch.zeros(T, 2, dtype=torch.float32, device=DEVICE)
        self._findTrajectory_step()

        DEBUG = True
        if DEBUG:
            try:
                init_traj_numpy = self.trajectory.detach().cpu().numpy()
                np.save('output/init_traj.npy', init_traj_numpy)
                # print("[Init] Initial trajectory saved to 'init_traj.npy'")
            except Exception as e:
                print(f"[Init Error] Failed to save init_traj.npy: {e}")

        return self.trajectory

    def _findPropParams_step(self):

        trajectory = self.trajectory
        propagation_params = self.propagation_params

        power_qt = self.feature_matrix[:, :, 0]
        angle_qt = self.feature_matrix[:, :, 1]
        delay_qt = self.feature_matrix[:, :, 2]
        
    ##### --- Initialize Constants ---
        L_qt = hard_em_utils.calculate_L_qt(trajectory, self.ap_locations)

        angle_qt_mean = hard_em_utils.calculate_angle_qt_mean(trajectory, self.ap_locations, self.ap_orientations)
        angle_qt1_mean = angle_qt_mean.unsqueeze(2)

    ##### --- Maximize 'Marginal Emission Probability Log Likelihood' until converge ---
        MAX_MEPLL_PropParams = -torch.inf

        while True:
            propagation_params_old = {k: v.clone() for k, v in propagation_params.items()}

            power_qk_average = (
                math_tools.calculate_weighted_average(
                    data=power_qt.unsqueeze(2),                 # [Q, T, 1]
                    weights=propagation_params['gamma_qtk'],    # [Q, T, K]
                    dim=1
                )
            )
            L_qk_average = (
                math_tools.calculate_weighted_average(
                    data=L_qt.unsqueeze(2),                     # [Q, T, 1]
                    weights=propagation_params['gamma_qtk'],    # [Q, T, K]
                    dim=1
                )
            )

        ##### --- Parameters Update related to Power ---
            propagation_params['alpha_qk'] = (
                hard_em_utils.calculate_alpha_qk(
                    power_qt, 
                    power_qk_average, 
                    L_qt, 
                    L_qk_average, 
                    propagation_params['gamma_qtk']
                )
            )
            propagation_params['beta_qk'] = (
                hard_em_utils.calculate_beta_qk(
                    propagation_params['alpha_qk'], 
                    power_qk_average, 
                    L_qk_average
                )
            )
            power_qtk_mean = (
                hard_em_utils.calculate_power_qtk_mean(
                    propagation_params['alpha_qk'], 
                    propagation_params['beta_qk'], 
                    L_qt
                )
            )
            propagation_params['power_qk_var'] = (
                hard_em_utils.calculate_power_qk_var(
                    power_qt,
                    power_qtk_mean,   
                    propagation_params['gamma_qtk']
                )
            ) 

            power_q1k_var = propagation_params['power_qk_var'].unsqueeze(1)
            
            power_distribution_qtk = (
                hard_em_utils.build_gaussian_distribution(
                    power_qtk_mean, 
                    power_q1k_var
                )
            )
            
        ##### --- Parameters Update related to Angle ---
            propagation_params['angle_k_var'] = (
                hard_em_utils.calculate_angle_k_var(
                    angle_qt_mean, 
                    angle_qt, 
                    propagation_params['gamma_qtk']
                )
            )
            angle_11k_var = propagation_params['angle_k_var'].view(1, 1, -1)

            angle_distribution_qtk = (
                hard_em_utils.build_gaussian_distribution(
                    angle_qt1_mean, 
                    angle_11k_var
                )
            )

        ##### --- Parameters Update related to Delay ---
            propagation_params['delay_k_mean'] = (
                hard_em_utils.calculate_delay_k_mean(
                    delay_qt, 
                    propagation_params['gamma_qtk']
                )
            )
            propagation_params['delay_k_var'] = (
                hard_em_utils.calculate_delay_k_var(
                    propagation_params['delay_k_mean'], 
                    delay_qt, 
                    propagation_params['gamma_qtk']
                )
            )
            
            delay_11k_mean = propagation_params['delay_k_mean'].view(1, 1, -1)
            delay_11k_var = propagation_params['delay_k_var'].view(1, 1, -1)
            
            delay_distribution_qtk = (
                hard_em_utils.build_gaussian_distribution(
                    delay_11k_mean, 
                    delay_11k_var
                )
            )
            
        ##### --- Parameters Update related to Global LOS ratio ---
            propagation_params['pi_k'] = hard_em_utils.calculate_pi(propagation_params['gamma_qtk'])

        ##### --- Calculate 'Marginal Emission Probability Log Likelihood' for PropParams ---
            MEPLL_PropParams_new = (
                hard_em_utils.calculate_MEPLL_PropParams(
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
                hard_em_utils.calculate_gamma_qtk(
                    propagation_params['pi_k'], 
                    power_distribution_qtk, 
                    angle_distribution_qtk, 
                    delay_distribution_qtk, 
                    power_qt, 
                    angle_qt, 
                    delay_qt
                )
            )

            DEBUG = False
            if DEBUG:
                print("===== Iteration =====\n")
                print(f"  MEPLL_New: {MEPLL_PropParams_new:.4f} vs Old_Max: {MAX_MEPLL_PropParams:.4f}")
                print(f"  Alpha (LOS/NLOS):\n{propagation_params['alpha_qk']}")
                print(f"  Beta (LOS/NLOS):\n{propagation_params['beta_qk']}")
                print(f"  Power Var (LOS/NLOS):\n{propagation_params['power_qk_var']}")
                print(f"  Angle Var (LOS/NLOS):\n{propagation_params['angle_k_var']}")
                print(f"  Delay Var (LOS/NLOS):\n{propagation_params['delay_k_var']}")
                print(f"  Global Pi (LOS/NLOS): {propagation_params['pi_k']}")
                print(f"  Gamma (LOS/NLOS): {propagation_params['gamma_qtk'][0]}")
            
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

        feature_matrix = self.feature_matrix
        config = self.config
        reference_grid = self.reference_grid
        propagation_params = self.propagation_params
        ap_locations = self.ap_locations

        T = self.num_sample
        G = self.reference_grid.shape[0]

    ##### --- Fisrt: Construct G * T Emission Probability step ---
        # Calculate emission probabilities for all grid points over time
        emission_probability_gt = (
            hard_em_utils.calculate_emission_probability(
                feature_matrix,
                reference_grid, 
                propagation_params,
                ap_locations, 
                self.ap_orientations, 
                DEVICE
            )
        )

    ##### --- Second: Viterbi Backpointer Updating step ---
        # Construct Neighbor Table
        G_index = torch.arange(G).to(DEVICE)
        G_neighbor_index_matrix = grid_tools.get_all_neighbor_indices(config, G_index, DEVICE) # shape: (G, 9)

        # Initialize Path Manager
        path_manager = ViterbiPathManager(G, T, reference_grid, DEVICE)

        # Initialize delta (log probability) with t=0 emission probs
        delta = torch.full((G,), -torch.inf, dtype=torch.float32, device=DEVICE)
        delta = emission_probability_gt[:, 0]

        # Main Loop: Update delta and backpointers from t=1 to T-1
        for t in range(1, T):
            current_emission_log_prob = emission_probability_gt[:, t]

            # Lazy Reconstruction: Retrieve history coords only if needed for Transformer
            history_coords = None
            if self.mode == 'TRANSFORMER':
                history_coords = path_manager.get_history_coords(end_t=t-1)

            # Find the best previous neighbor and max transition probability
            G_winner_neighbor_index, max_value = (
                hard_em_utils.get_winner_neighbor_info(
                    G_neighbor_index_matrix, 
                    delta,
                    history_coords,
                    self.model,    
                    self.mode, 
                    self.SOS_TOKEN
                )
            )

            # Update delta with current emission and best transition
            delta = current_emission_log_prob + max_value

            # Record the best parent indices for backtracking
            path_manager.update(t, G_winner_neighbor_index)

        # Find the end node with maximum probability
        MEPLL_Trajectory, best_end_node = torch.max(delta, dim=0)

        # Traceback the optimal path using backpointers
        self.trajectory = path_manager.traceback_final_trajectory(T-1, best_end_node)
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
