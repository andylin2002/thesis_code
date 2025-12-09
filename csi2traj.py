import utils
from typing import Dict, Any, Optional
import torch
import numpy as np

def run_csi2traj(
        config: Dict[str, Any], 
        reference_grid: torch.Tensor, 
        model: Optional[torch.nn.Module], 
        mode: str, 
        directions_vectors: np.ndarray, 
        raw_csi_data: torch.Tensor
    ) -> Optional[torch.Tensor]:

##### --- Importing Raw CSI Data ---
    if raw_csi_data is None:
        print("CSI data loading failed.")
        return

##### --- Sanitize Input: Replace NaN/Inf with 0.0 ---
    raw_csi_data = torch.nan_to_num(raw_csi_data, nan=0.0, posinf=0.0, neginf=0.0)
    
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
        config=config,
        reference_grid=reference_grid,
        model=model, 
        mode=mode, 
        directions_vectors=directions_vectors
    )

##### --- Predicted Trajectory ---

    return trajectory
