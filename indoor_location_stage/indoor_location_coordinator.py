from typing import Dict, Any, Optional
import torch

from .EM_algorithm.em_core import EM_Algorithm

TypeTrajectory = torch.Tensor

def run_indoor_location(
        feature_matrix: torch.Tensor, 
        config: Dict[str, Any],
        reference_grid: torch.Tensor, 
        context: Dict[str, Any],
        model: Optional[torch.nn.Module], 
        mode: str
    ) -> Optional[TypeTrajectory]:

##### --- EM Algorithm ---

    em_engine = EM_Algorithm(
        feature_matrix=feature_matrix, 
        config=config, 
        reference_grid=reference_grid,
        context=context, 
        model=model, 
        mode=mode
    )

    trajectory = em_engine.run_em_iterations()

    return trajectory