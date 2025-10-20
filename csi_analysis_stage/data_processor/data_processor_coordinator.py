import numpy as np
from typing import Dict, Any, Optional

from . import packeting
from . import aggregation

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_data_processor(
        raw_csi_data: np.ndarray, 
        config: Dict[str, Any]
) -> Optional[np.ndarray]:
    
##### --- Parameters ---
    ap_data = config.get('ACCESS_POINTS', {})
    num_ap = len(ap_data)
    num_sample = config['NUM_SAMPLE']
    num_packet = config['NUM_PACKET']

##### --- Put Raw CSI Data on GPU ---
    raw_csi_data_tensor = torch.from_numpy(raw_csi_data).to(DEVICE).to(torch.complex64) # (TP, Q, N, M)
    raw_csi_data_permuted_tensor = raw_csi_data_tensor.permute(1, 0, 2, 3).contiguous() # (Q, TP, N, M)
    
##### --- Packeting ---
    # input: (Q, TP, N, M)   output: (Q, T, P, N, M)
    packeted_csi_gpu = packeting.run_packeting_gpu(
        csi_data=raw_csi_data_permuted_tensor,
        T_time=num_sample,
        P_packet=num_packet
    )

    num_batch = num_sample * num_ap
    non_aggregated_csi_gpu = packeted_csi_gpu.reshape(
        num_batch, 
        num_packet, 
        *packeted_csi_gpu.shape[3:]
    ) # shape: (B=QT, P, N, M)

##### --- Aggregation ---
    # input: (QT, P, N, M)   output: (QT, 1, N, M)
    aggregated_csi_gpu = aggregation.run_aggregation_gpu(non_aggregated_csi_gpu)

    # input: (QT, 1, N, M)   output: (T, Q, 1, N, M)
    reshape_aggregated_csi_gpu = aggregated_csi_gpu.reshape(
        num_sample, 
        num_ap, 
        1, 
        *aggregated_csi_gpu.shape[2:]
    )

    processed_csi_tensor = reshape_aggregated_csi_gpu.squeeze(2) # shape: (T, Q, N, M)
    processed_csi_numpy = processed_csi_tensor.cpu().numpy()

    return processed_csi_numpy