import utils
from typing import Dict, Any, Optional
import torch

class CSItoTRAJ:
    def __init__(
            self, 
            config: Dict[str, Any], 
            reference_grid: torch.Tensor
        ):
        
        self.config = config
        self.reference_grid = reference_grid

    def run_csi2traj(self, context: Dict[str, Any]) -> Optional[torch.Tensor]:

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
            config = self.config,
            reference_grid=self.reference_grid,
            context=context
            )

    ##### --- Predicted Trajectory ---
        print("at csi2traj.py: ", trajectory[0:10])

        return trajectory
