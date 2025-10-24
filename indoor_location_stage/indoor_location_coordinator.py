import numpy as np
from typing import Dict, Any, Optional
import torch

from .EM_algorithm.em_core import EM_Algorithm

TypeTrajectory = torch.Tensor

def run_indoor_location(
    feature_matrix: torch.Tensor, 
    reference_grid: torch.Tensor, 
    APs_LOS_ratio: torch.Tensor,
    config: Dict[str, Any]
) -> Optional[TypeTrajectory]:

##### --- EM Algorithm ---

    em_engine = EM_Algorithm(
        feature_matrix=feature_matrix, 
        reference_grid=reference_grid, 
        gamma_APs_LOS_ratio=APs_LOS_ratio,
        config=config
    )

    trajectory = em_engine.run_em_iterations()

    return trajectory