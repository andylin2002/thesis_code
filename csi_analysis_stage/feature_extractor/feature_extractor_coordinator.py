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
    num_batch = num_ap * num_sample
    batch_input_csi = processed_csi.reshape(num_batch, *processed_csi.shape[2:]).contiguous() # (QT, N, M)
    
##### --- Power ---
    power_tensor_flat = power_extractor.extract_power_batch(
        input_csi=batch_input_csi
    )

##### --- MMP algorithm (AoA & ToF) ---
    mmp_engine = MMP_Algorithm(config=config)

    angle_tensor_flat, tof_tensor, eigv_x, eigv_y = mmp_engine.estimate_aoa_tof_batch(
        input_csi=batch_input_csi
    )

##### --- Delay ---
    if config['USE_BASELINE']:
        magnitude_csi = batch_input_csi.abs()
        variance_tensor = torch.var(magnitude_csi.reshape(num_batch, -1), dim=1)
        delay_tensor_flat = 10 * torch.log10(variance_tensor + 1e-9)
    else:   # MYMETHOD
        delay_tensor_flat = (
            delay_estimator.estimate_delay_batch(
                batch_input_csi, 
                tof_tensor, 
                eigv_x, 
                eigv_y
            )
        )

##### --- Stacking & Reshape ---
    features_stacked_flat = torch.stack([
        power_tensor_flat, 
        angle_tensor_flat, 
        delay_tensor_flat
    ], dim=1)

    feature_matrix_tensor = features_stacked_flat.reshape(num_ap, num_sample, 3)

    return feature_matrix_tensor