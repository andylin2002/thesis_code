import numpy as np
from typing import Dict, Any, Optional

from .data_processor import run_data_processor
from .feature_extractor import run_feature_extractor

def run_csi_analysis(
        raw_csi_data: np.ndarray, 
        config: Dict[str, Any]
) -> Optional[np.ndarray]:

##### --- Data Preprocessing ---

    ## 希望所有AP用GPU同時計算

    processed_csi = run_data_processor(
        raw_csi_data=raw_csi_data,
        config=config)

##### --- Feature Extraction ---

    feature_matrix = run_feature_extractor(
        processed_csi=processed_csi,
        config=config
    )

    return feature_matrix
    