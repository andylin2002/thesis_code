import numpy as np
from typing import Dict, Any, Optional

from .EM_algorithm.em_core import EM_Algorithm

PredictedPath = np.ndarray

def run_indoor_location(
    feature_matrix: np.ndarray, 
    reference_grid: np.ndarray, 
    config: Dict[str, Any]
) -> Optional[PredictedPath]:
    
    num_sample = config['NUM_SAMPLE']

##### --- EM Algorithm ---

    em_engine = EM_Algorithm(
        feature_matrix=feature_matrix, 
        reference_grid=reference_grid, 
        config=config
    )

    predicted_path = em_engine.run_em_iterations()
    if predicted_path is None:
        print("Error: EM algorithm failed to produce a path.")
        return np.zeros((num_sample, 2)) 

    return predicted_path