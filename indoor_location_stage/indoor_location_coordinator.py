import numpy as np
from typing import Dict, Any, Optional, Tuple

def run_indoor_location(
    feature_matrix: np.ndarray, 
    reference_grid: np.ndarray, 
    config: Dict[str, Any]
) -> Optional[np.ndarray]:
    
    num_sample = config['NUM_SAMPLE']
    predicted_path = predicted_path = np.zeros((num_sample, 2))

##### --- Initializing EM parameters ---

    return predicted_path