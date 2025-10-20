# csi_analysis_stage/data_processor/packeting.py

import torch
from typing import Dict, Any, Optional

def run_packeting_gpu(csi_data: torch.Tensor, T_time: int, P_packet: int) -> torch.Tensor:
    Q_AP, TP_TOTAL, N_ANT, M_SUB = csi_data.shape
    
    # Check if the input N_TIME (1500) matches T*P (100*15)
    if TP_TOTAL != T_time * P_packet:
        raise ValueError(f"[PACKETING] Total time dimension ({TP_TOTAL}) does not match T*P ({T_time * P_packet}).")

    # Reshape the N_TIME dimension (index 1) into (T, P)
    # The output shape is (Q, T, P, N, M)
    packeted_data = csi_data.reshape(
        Q_AP, 
        T_time, 
        P_packet, 
        N_ANT, 
        M_SUB
    )

    return packeted_data