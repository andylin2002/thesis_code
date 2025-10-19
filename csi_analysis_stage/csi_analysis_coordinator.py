import numpy as np
from typing import Dict, Any, Optional



def run_csi_analysis(
        raw_csi_data: np.ndarray, 
        env_config: Dict[str, Any], 
        sys_config: Dict[str, Any]
) -> Optional[np.ndarray]:
    
##### --- Load Static Parameters ---
    ap_data = env_config.get('ACCESS_POINTS', {})
    num_ap = len(ap_data)
    num_sample = sys_config['NUM_SAMPLE']
    feature_matrix = np.zeros((num_sample, num_ap, 3), dtype=np.float32) # 3 means power, angle, delay

##### --- Data Preprocessing ---



##### --- Feature Extraction ---

    return feature_matrix
    