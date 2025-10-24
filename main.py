import utils
import torch
import argparse
from csi2traj import CSItoTRAJ

CONFIG_PATH = 'config.yaml'

def main():
    parser=argparse.ArgumentParser(description='CSI Indoor Position System Parameter')
    parser.add_argument('--em_max_iter', type=int, default=20)

    args=parser.parse_args()

##### --- Loading Environment & System Configuration ---
    config = utils.load_yaml_config(CONFIG_PATH)
    if not (config):
        print("Configuration loading failed.")
        return
    
##### --- Put Hyperparameter into Config ---
    config['EM_MAX_ITER'] = args.em_max_iter

##### --- Reference Point Setup ---
    reference_grid, x_bounds, y_bounds = utils.generate_reference_grid(config)

    APs_LOS_ratio = torch.full((4, 2), 0.5, dtype=torch.float32)

    csi2traj_engine = CSItoTRAJ(config, reference_grid, APs_LOS_ratio)

    for i in range(1):
        csi2traj_engine.run_csi2traj()


if __name__ == '__main__':
    main()