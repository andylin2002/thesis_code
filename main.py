# main.py

import utils
import numpy as np

ENV_CONFIG_PATH = 'config/env_config.yaml'
SYS_CONFIG_PATH = 'config/system_config.yaml'

def main():
##### --- Loading Environment & System Configuration ---
    env_config = utils.load_yaml_config(ENV_CONFIG_PATH)
    sys_config = utils.load_yaml_config(SYS_CONFIG_PATH)
    if not (env_config and sys_config):
        print("Configuration loading failed.")
        return

##### --- Reference Point Setup ---
    reference_grid, x_bounds, y_bounds, ap_locations = utils.generate_reference_grid(env_config)

##### --- Importing Raw CSI Data ---
    RAW_CSI_PATH = 'csi_sample.npy'
    raw_csi_data = utils.load_raw_csi(RAW_CSI_PATH)

    if raw_csi_data is None:
        print("CSI data loading failed.")
        return
    
##### --- Starting CSI Analysis Stage ---
    from csi_analysis_stage import run_csi_analysis

    feature_matrix = run_csi_analysis(
        raw_csi_data=raw_csi_data,
        env_config=env_config,
        sys_config=sys_config
    )

##### --- Starting Indoor Location Stage ---
    from indoor_location_stage import run_indoor_location

    predicted_path = run_indoor_location(
    feature_matrix=feature_matrix, 
    reference_grid=reference_grid, 
    env_config=env_config,
    sys_config=sys_config
    )

    print(predicted_path)

if __name__ == '__main__':
    main()