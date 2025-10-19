import numpy as np
from typing import Dict, Any, Optional

from . import mmp_core
from . import power_extractor
from . import delay_estimator

def run_feature_extractor(
        processed_csi: np.ndarray, 
        config: Dict[str, Any]
) -> Optional[np.ndarray]:
    
    return 0#feature_matrix