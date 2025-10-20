import numpy as np
from typing import Dict, Any, Optional

from . import mmp_core
from . import power_extractor
from . import delay_estimator

def run_feature_extractor(
        processed_csi: np.ndarray, 
        config: Dict[str, Any]
) -> Optional[np.ndarray]:
    
##### --- Parameters ---
    ap_data = config.get('ACCESS_POINTS', {})
    num_ap = len(ap_data)
    num_sample = config['NUM_SAMPLE']

    feature_matrix = np.zeros((num_sample, num_ap, 3), dtype=np.float32)
    

    return 0#feature_matrix