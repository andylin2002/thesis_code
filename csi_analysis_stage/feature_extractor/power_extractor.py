import torch
from typing import Optional

def extract_power_batch(input_csi: torch.Tensor) -> Optional[torch.Tensor]:

    # (TODO)
    
##### --- Square Every Element of CSI Data ---
    magnitude_squared = input_csi.abs().pow(2)
    
##### --- Sum ---
    sum_of_powers = magnitude_squared.sum(dim=[1, 2])

##### --- Avoid log value to be negative infinity ---
    sum_of_powers = torch.clamp(sum_of_powers, min=1e-9)

##### --- Get Power Value ---
    power_tensor_flat = 10.0 * torch.log10(sum_of_powers)
    
    return power_tensor_flat