# markov_model/uniform_markov.py (修正後的完整程式碼)

from typing import Dict, Any, Optional
import torch
import math

def generate_uniform_markov_trajectory(
    config: Dict[str, Any], 
    reference_grid: torch.Tensor, 
    start_point: torch.Tensor,  # 🚨 接收二維座標 Tensor
    device: torch.device,
) -> torch.Tensor:
    """
    使用 Uniform Markov Model 生成一條連續的路徑 (純 GPU Tensor 操作)。
    """

    # --- 1. 網格參數計算 (W, H, T) ---
    x_min = config['X_BOUNDS'][0]
    x_max = config['X_BOUNDS'][1]
    resolution = config['GRID_RESOLUTION_M']
    grid_intervals = (x_max - x_min) / resolution
    grid_width = int(round(grid_intervals)) + 1
    
    G = reference_grid.shape[0]
    W = grid_width
    H = G // W 
    T = config['NUM_SAMPLE']
    
    trajectory = torch.zeros(T, 2, dtype=torch.float32, device=device)
    
    # --- 2. 🚨 核心修正點：確保 start_point 為 (2,) shape 並轉換為索引 ---
    
    # 檢查並強制將 start_point 轉換為 (2,) shape，防止 (1, 2) 帶來的計算錯誤
    if start_point.dim() > 1 and start_point.shape[0] == 1:
        start_point = start_point.squeeze(0) # 從 (1, 2) 變為 (2,)
    elif start_point.dim() > 1 and start_point.numel() == 2:
        start_point = start_point.flatten() # 處理其他可能的多餘維度
    
    # reference_grid (G, 2) - start_point.unsqueeze(0) (1, 2) = (G, 2)
    distances = torch.linalg.norm(
        reference_grid - start_point.unsqueeze(0).to(reference_grid.device), 
        dim=1
    )
    
    # 找到最近網格點的索引 (這必須是 start_point 本身的索引)
    start_grid_idx = torch.argmin(distances).item()

    # 將起始索引轉換為 Tensor，用於迴圈迭代
    current_idx_tensor = torch.tensor([start_grid_idx], dtype=torch.long, device=device)
    
    # --- 3. 馬可夫鏈運算 ---
    
    # 預先計算所有可能的行偏移量和列偏移量
    delta_row = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)
    delta_col = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.long, device=device)
    
    for t in range(T):
        # 1. 記錄當前位置 (t=0 時，trajectory[0] = reference_grid[start_grid_idx] = start_point 座標)
        trajectory[t] = reference_grid[current_idx_tensor[0]]
        
        # 2. 計算當前一維索引的 (row, col)
        current_row = current_idx_tensor[0] // W
        current_col = current_idx_tensor[0] % W
        
        # 3. 計算所有 9 個候選點的 (row, col)
        next_row_candidates = current_row + delta_row
        next_col_candidates = current_col + delta_col
        
        # 4. 邊界檢查 (使用掩碼)
        mask_row_valid = (next_row_candidates >= 0) & (next_row_candidates < H)
        mask_col_valid = (next_col_candidates >= 0) & (next_col_candidates < W)
        valid_mask = mask_row_valid & mask_col_valid
        
        # 過濾出有效的鄰居索引
        valid_indices = torch.where(valid_mask)[0]
        
        if valid_indices.numel() == 0:
            break
            
        # 5. 均勻隨機選擇下一個點
        perm = torch.randperm(valid_indices.numel(), device=device)
        chosen_global_index_in_delta = valid_indices[perm[0]]

        # 獲取選中的下一個 (row, col)
        next_row = delta_row[chosen_global_index_in_delta] + current_row
        next_col = delta_col[chosen_global_index_in_delta] + current_col
        
        # 6. 轉換回一維索引並更新
        next_idx = next_row * W + next_col
        current_idx_tensor[0] = next_idx
        
    return trajectory