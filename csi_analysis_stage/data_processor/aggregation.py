# csi_analysis_stage/data_processor/aggregation.py

import torch
from typing import Optional

def run_aggregation_gpu(non_aggregated_csi_gpu: torch.Tensor) -> Optional[torch.Tensor]:
   
##### --- Parameter Setup ---
    B_batch, P_packet, N_antenna, M_subcarrier = non_aggregated_csi_gpu.shape

##### --- Reshape ---
    combined_tensor = non_aggregated_csi_gpu.view(B_batch, P_packet, N_antenna * M_subcarrier)
    svd_input_batch = combined_tensor.permute(0, 2, 1).contiguous()

##### --- SVD decomposition ---
    U, S, Vh = torch.linalg.svd(svd_input_batch)
    first_singular_vector = U[:, :, 0]
    max_singular_value = S[:, 0]

##### --- Get Aggregated CSI data ---
    aggregated_output_flat = first_singular_vector * max_singular_value.unsqueeze(1)
    aggregated_output = aggregated_output_flat.view(B_batch, 1, N_antenna, M_subcarrier)

    return aggregated_output