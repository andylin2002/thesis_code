import utils
import numpy as np
from typing import Dict, Any, Optional
import torch

class CSItoTRAJ:
    def __init__(
            self, 
            config: Dict[str, Any], 
            reference_grid: torch.Tensor, 
            APs_LOS_ratio: torch.Tensor
        ):
        
        self.config = config
        self.reference_grid = reference_grid
        self.APsLOS_ratio = APs_LOS_ratio

    def run_csi2traj(self):

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
            config = self.config)

    ##### --- Starting Indoor Location Stage ---
        from indoor_location_stage import run_indoor_location

        trajectory = run_indoor_location(
            feature_matrix=feature_matrix,
            reference_grid=self.reference_grid,
            APs_LOS_ratio=self.APsLOS_ratio,
            config = self.config)

    ##### --- Predicted Trajectory ---
        print("at csi2traj.py: ", trajectory[0:10])
