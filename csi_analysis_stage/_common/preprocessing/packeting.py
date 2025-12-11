# csi_analysis_stage/data_processor/packeting.py

import torch
from typing import Dict, Any, Optional

def run_packeting_gpu(csi_data: torch.Tensor, T_time: int, P_packet: int) -> torch.Tensor:
    Q_ap, TP_total, N_antenna, M_subcarrier = csi_data.shape
    
##### --- Check if the input N_TIME (1500) matches T*P (100*15) ---
    if TP_total != T_time * P_packet:
        raise ValueError(f"[PACKETING] Total time dimension ({TP_total}) does not match T*P ({T_time * P_packet}).")

##### Reshape the N_TIME dimension (index 1) into (T, P)
    # The output shape is (Q, T, P, N, M)
    packeted_data = csi_data.reshape(
        Q_ap, 
        T_time, 
        P_packet, 
        N_antenna, 
        M_subcarrier
    )

    return packeted_data