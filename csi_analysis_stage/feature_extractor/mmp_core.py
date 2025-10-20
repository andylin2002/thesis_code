# csi_analysis_stage/feature_extractor/mmp_core.py

import torch
from typing import Dict, Any, Optional, Tuple, List

def estimate_aoa_tof_batch(
    batch_input_csi: torch.Tensor, 
    config: Dict[str, Any]
) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    
    # (TODO)
    
    print(f"      [MMP] Input Shape: {batch_input_csi.shape}")

    N_batches = batch_input_csi.shape[0]

    # --- 1. AoA 處理 (不變) ---
    # AoA 只需要一個值，所以維持 (N_batches,) 的矩形張量
    aoa_tensor_flat = torch.rand(N_batches, device=batch_input_csi.device) * 360 

    # --- 2. ToF 處理 (關鍵變動：List[Tensor]) ---
    
    # 模擬 400 個批次，每個批次有不同的 ToF 集合長度 L(t,q)
    tof_tensor_list: List[torch.Tensor] = []
    
    # 這裡的迴圈是必須的，因為我們正在模擬計算 L(t,q)
    for i in range(N_batches):
        # L(t,q) 模擬長度 (例如 5 到 15 個多徑分量)
        L_i = torch.randint(low=5, high=15, size=(1,)).item()
        
        # 創建一個長度為 L_i 的 ToF 集合 (e.g., 1e-9 to 1e-7 s)
        tof_set = torch.rand(L_i, device=batch_input_csi.device) * 1e-7
        tof_tensor_list.append(tof_set)
        
    print(f"      [MMP] AoA Output Shape: {aoa_tensor_flat.shape}")
    print(f"      [MMP] ToF Output Type: List[Tensor] (Length {len(tof_tensor_list)})")
    
    # 注意：現在 ToF 回傳一個 List[Tensor]
    return aoa_tensor_flat, tof_tensor_list