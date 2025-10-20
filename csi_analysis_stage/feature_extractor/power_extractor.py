# csi_analysis_stage/feature_extractor/power_extractor.py

import torch
from typing import Optional

def extract_power_batch(batch_input_csi: torch.Tensor) -> Optional[torch.Tensor]:

    # (TODO)
    
    print(f"      [POWER] Input Shape: {batch_input_csi.shape}")
    
    # 1. Compute magnitude squared: |H|^2 
    magnitude_squared = batch_input_csi.abs().pow(2)
    
    # 2. Sum over N_ant and N_sub dimensions (indices 1 and 2)
    # Resulting shape: (N_batches,)
    power_tensor_flat = magnitude_squared.sum(dim=[1, 2])
    
    print(f"      [POWER] Output Shape: {power_tensor_flat.shape}")
    return power_tensor_flat