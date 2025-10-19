import numpy as np
from typing import Dict, Any, Optional

from . import packeting
from . import aggregation

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_data_processor(
        raw_csi_data: np.ndarray, 
        config: Dict[str, Any]
) -> Optional[np.ndarray]:
    
    ap_data = config.get('ACCESS_POINTS', {})
    num_ap = len(ap_data)
    num_sample = config['NUM_SAMPLE']
    num_packet = config['NUM_PACKET']

    print(num_ap)
    
    return 0#processed_csi