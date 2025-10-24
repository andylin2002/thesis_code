import numpy as np
from typing import Dict, Any, Optional

from .data_processor import run_data_processor
from .feature_extractor import run_feature_extractor

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_csi_analysis(
        raw_csi_data: np.ndarray, 
        config: Dict[str, Any]
) -> Optional[torch.Tensor]:
    
##### --- Put CSI Data on GPU ---
    raw_csi_data_tensor = torch.from_numpy(raw_csi_data).to(DEVICE).to(torch.complex64) # (Q, TP, N, M)

##### --- Data Preprocessing (on GPU) ---

    processed_csi_tensor = run_data_processor(
        raw_csi_data_tensor=raw_csi_data_tensor,
        config=config
    )

##### --- Feature Extraction (on GPU) ---

    feature_matrix_tensor = run_feature_extractor(
        processed_csi=processed_csi_tensor,
        config=config
    )

    return feature_matrix_tensor
    