from typing import Dict, Any, Optional

from . import packeting
from . import aggregation

import torch

def run_data_processor(
        raw_csi_data_tensor: torch.Tensor, 
        config: Dict[str, Any]
) -> Optional[torch.Tensor]:
    
##### --- Parameters ---
    ap_data = config.get('ACCESS_POINTS', {})
    num_ap = len(ap_data)
    num_sample = config['NUM_SAMPLE']
    num_packet = config['NUM_PACKET']

##### --- Permutation ---
    raw_csi_data_permuted_tensor = raw_csi_data_tensor.permute(1, 0, 2, 3).contiguous() # (Q, TP, N, M)
    
##### --- Packeting ---
    # input: (Q, TP, N, M)
    packeted_csi_gpu = packeting.run_packeting_gpu(
        csi_data=raw_csi_data_permuted_tensor,
        T_time=num_sample,
        P_packet=num_packet
    ) # output: (Q, T, P, N, M)

    num_batch = num_sample * num_ap
    non_aggregated_csi_gpu = packeted_csi_gpu.reshape(
        num_batch, 
        num_packet, 
        *packeted_csi_gpu.shape[3:]
    ) # shape: (B=QT, P, N, M)

##### --- Aggregation ---
    # input: (QT, P, N, M)
    aggregated_csi_gpu = aggregation.run_aggregation_gpu(non_aggregated_csi_gpu)
    # output: (QT, 1, N, M)

    # input: (QT, 1, N, M)
    reshape_aggregated_csi_gpu = aggregated_csi_gpu.reshape(
        num_sample, 
        num_ap, 
        1, 
        *aggregated_csi_gpu.shape[2:]
    ) # output: (T, Q, 1, N, M)

    processed_csi_tensor = reshape_aggregated_csi_gpu.squeeze(2) # shape: (T, Q, N, M)

    return processed_csi_tensor