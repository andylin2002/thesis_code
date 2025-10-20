# csi_analysis_stage/feature_extractor/delay_estimator.py

import torch
from typing import Optional, List

def estimate_delay_batch(tof_tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
    
    # (TODO)
    
    print(f"      [DELAY] Input Type: List[Tensor] (Length {len(tof_tensor_list)})")
    
    N_batches = len(tof_tensor_list)
    delay_spreads = torch.zeros(N_batches, device=tof_tensor_list[0].device)
    
    # --- 核心計算 (迴圈計算標準差) ---
    # NOTE: 在 PyTorch 中，計算 List[Tensor] 上的標準差無法完全向量化，
    #       需要使用 Python 迴圈來執行每個 Tensor 的 std() 操作。
    
    for i, tof_set in enumerate(tof_tensor_list):
        if tof_set.numel() > 1:
            # 計算該 ToF 集合的標準差 (Delay Spread τ)
            # .std() 會自動在 GPU 上執行
            delay_spreads[i] = tof_set.std()
        else:
            # 如果集合只有一個點，標準差為 0
            delay_spreads[i] = torch.tensor(0.0, device=tof_set.device)

    print(f"      [DELAY] Output Shape: {delay_spreads.shape} (N_batches,)")
    return delay_spreads