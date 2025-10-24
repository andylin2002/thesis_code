import utils
import numpy as np

CONFIG_PATH = 'config.yaml'

def main():
##### --- Loading Environment & System Configuration ---
    config = utils.load_yaml_config(CONFIG_PATH)
    if not (config):
        print("Configuration loading failed.")
        return

##### --- Reference Point Setup ---
    reference_grid, x_bounds, y_bounds, ap_locations = utils.generate_reference_grid(config)

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
        config=config
    )

##### --- Starting Indoor Location Stage ---
    from indoor_location_stage import run_indoor_location

    trajectory = run_indoor_location(
    feature_matrix=feature_matrix, 
    reference_grid=reference_grid, 
    APs_LOS_ratio=APs_LOS_ratio, # (TODO: main -> csi2traj and pass gamma with iteration in new 'main.py')
    config=config
    )

##### --- Predicted Trajectory ---
    print(trajectory[0:10])

if __name__ == '__main__':
    main()