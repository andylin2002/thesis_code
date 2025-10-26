import utils
import torch
import argparse
from csi2traj import CSItoTRAJ

CONFIG_PATH = 'config.yaml'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser=argparse.ArgumentParser(description='CSI Indoor Position System Parameter')
    parser.add_argument('--em_max_iter', type=int, default=20)
    parser.add_argument('--em_findPropParams_threshold', type=float, default=1e-2)

    args=parser.parse_args()

##### --- Loading Environment & System Configuration ---
    config = utils.load_yaml_config(CONFIG_PATH)
    if not (config):
        print("Configuration loading failed.")
        return
    
##### --- Put Hyperparameter into Config ---
    config['EM_MAX_ITER'] = args.em_max_iter
    config['EM_M_STEP_TH'] = args.em_findPropParams_threshold

##### --- Reference Point Setup ---
    reference_grid, x_bounds, y_bounds = utils.generate_reference_grid(config)

##### --- Dynamic parameters ---
    context = {
            'current_round': None,
            'last_predicted_point': None, 
            'APs_LOS_ratio': None
        }

##### --- LOS/NLOS ration for each AP --- 

    ap_data = config.get('ACCESS_POINTS', {})
    Q = len(ap_data)
    T = config['NUM_SAMPLE']

    epsilon = 1e-6
    random_noise = torch.rand(Q, T, 2, device=DEVICE) * epsilon

    APs_LOS_ratio_symmetric = torch.full((Q, T, 2), 0.5, dtype=torch.float32, device=DEVICE)
    APs_LOS_ratio = APs_LOS_ratio_symmetric + random_noise

    context['APs_LOS_ratio'] = APs_LOS_ratio

##### --- Dummy last_predicted_point ---
    context['last_predicted_point'] = torch.zeros(1, 2, dtype=torch.float32, device=DEVICE)

##### --- Implement CSItoTRAJ ---
    csi2traj_engine = CSItoTRAJ(config, reference_grid)

    for round in range(1):
        context['current_round'] = round

        trajectory = csi2traj_engine.run_csi2traj(context)

        context['last_predicted_point'] = trajectory[-1:].clone().detach()

##### --- Transformer Training --- (TODO)


if __name__ == '__main__':
    main()