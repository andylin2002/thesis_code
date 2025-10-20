# csi_analysis_stage/data_processor/aggregation.py

import torch
from typing import Optional

def run_aggregation_gpu(non_aggregated_csi_gpu: torch.Tensor) -> Optional[torch.Tensor]:
   
    # (TODO) SVD aggregation!
    aggregated_output = torch.mean(non_aggregated_csi_gpu, dim=1, keepdim=True)

    return aggregated_output