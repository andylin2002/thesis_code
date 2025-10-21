from typing import Dict, Any, Optional

from .mmp_core import MMP_Algorithm
from . import power_extractor
from . import delay_estimator

import torch

def run_feature_extractor(
        processed_csi: torch.Tensor, 
        config: Dict[str, Any]
) -> Optional[torch.Tensor]:
    
##### --- Parameters ---
    ap_data = config.get('ACCESS_POINTS', {})
    num_ap = len(ap_data)
    num_sample = config['NUM_SAMPLE']

##### --- Prepare for Batch Processing ---
    num_batch = num_sample * num_ap
    batch_input_csi = processed_csi.reshape(num_batch, *processed_csi.shape[2:]).contiguous() # (QT, N, M)
    
##### --- Power ---
    power_tensor_flat = power_extractor.extract_power_batch(
        batch_input_csi=batch_input_csi
    )

##### --- MMP algorithm (AoA & ToF) ---
    mmp_engine = MMP_Algorithm(config=config)

    angle_tensor_flat, tof_tensor_flat = mmp_engine.estimate_aoa_tof_batch(
        batch_input_csi=batch_input_csi
    )

##### --- Delay ---
    delay_tensor_flat = delay_estimator.estimate_delay_batch(tof_tensor_flat)

##### --- Stacking & Reshape ---
    features_stacked_flat = torch.stack([
        power_tensor_flat, 
        angle_tensor_flat, 
        delay_tensor_flat
    ], dim=1) 

    feature_matrix_tensor = features_stacked_flat.reshape(num_sample, num_ap, 3)

    return feature_matrix_tensor