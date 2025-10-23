# csi_analysis_stage/feature_extractor/delay_estimator.py

import torch
from typing import Tuple
import torch.nn.functional as F

def estimate_delay_batch(
        tof_tensor: torch.Tensor, 
        eigv_x: torch.Tensor, 
        eigv_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    weight_tensor = F.softmax(abs(eigv_x) * abs(eigv_y), dim=1)
    adjusted_tof_tensor = weight_tensor * tof_tensor

##### --- Mean of Delay ---
    mean_delay = adjusted_tof_tensor.sum(dim=1, keepdim=True)

##### --- Weighted Standard Deviation of Delay ---

    diff_squared = (tof_tensor - mean_delay).pow(2)
    weighted_variance = weight_tensor * diff_squared
    std_dev_delay = torch.sqrt(weighted_variance.sum(dim=1, keepdim=True))
    
##### --- Unit Conversion ---
    log10_std_dev = torch.log10(std_dev_delay).squeeze()
    delay_tensor_flat = 10 * log10_std_dev

    return delay_tensor_flat