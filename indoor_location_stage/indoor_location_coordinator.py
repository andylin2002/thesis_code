from typing import Dict, Any, Optional
import torch
import numpy as np

from .EM_algorithm.em_core import EM_Algorithm

TypeTrajectory = torch.Tensor

def run_indoor_location(
        feature_matrix: torch.Tensor, 
        config: Dict[str, Any],
        reference_grid: torch.Tensor, 
        model: Optional[torch.nn.Module], 
        mode: str, 
        directions_vectors: np.ndarray
    ) -> Optional[TypeTrajectory]:

##### --- EM Algorithm ---

    em_engine = EM_Algorithm(
        feature_matrix=feature_matrix, 
        config=config, 
        reference_grid=reference_grid, 
        model=model, 
        mode=mode, 
        directions_vectors=directions_vectors
    )

    trajectory = em_engine.run_em_iterations()

    return trajectory