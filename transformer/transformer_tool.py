import torch
import numpy as np
import scipy.spatial.distance
from typing import Dict, Any, Tuple
from .model import QuantizedTF
import torch.nn as nn
import torch.nn.functional as F

from transformer.architecture.batch import subsequent_mask

def create_transformer_instance(
    config: Dict[str, Any], 
    N_DIRECTIONS: int, 
    device: torch.device
):
    
    D_MODEL = config['EMB_SIZE']
    N_HEADS = config['HEADS']
    N_LAYERS = config['LAYERS']
    DROPOUT = config['DROPOUT']
    D_FF = config['D_FF']

    model = QuantizedTF(
                enc_inp_size=2, 
                dec_inp_size=(N_DIRECTIONS + 1),
                dec_out_size=N_DIRECTIONS,
                
                layer=N_LAYERS,
                d_model=D_MODEL,
                d_ff=D_FF,
                h=N_HEADS,
                dropout=DROPOUT
            ).to(device)
    
    return model

def convert_long_trajectory_to_ids(
    batch_traj: torch.Tensor, directions_vectors: np.ndarray, config: Dict[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    T = batch_traj.shape[1]
    B = batch_traj.shape[0]

    inp_coords = batch_traj[:, :(T - 1), :].clone()
    
    velocities_abs = batch_traj[:, 1:, :] - batch_traj[:, :-1, :]
    velocities_flat = velocities_abs.cpu().reshape(-1, 2).numpy()
    target_ids_flat = scipy.spatial.distance.cdist(velocities_flat, directions_vectors).argmin(axis=1)
    target_ids = torch.tensor(target_ids_flat, dtype=torch.long, device=device).reshape(B, (T - 1))

    return inp_coords, target_ids

def generate_transformer_trajectory(
    model: torch.nn.Module,
    start_point: torch.Tensor, 
    directions_vectors: np.ndarray, 
    T_length: int, 
    SOS_TOKEN: int, 
    device: torch.device
) -> torch.Tensor:
    
    # 1. 初始化軌跡：將起始點 P1 作為序列的第一個元素
    # Shape: [1, 1, 2] (Batch size 1, 1 step, 2D)
    trajectory_coords = start_point.clone().unsqueeze(0) 
    
    # 確保模型和環境處於正確狀態
    model.eval()
    
    with torch.no_grad():
        
        # 2. 初始化 Decoder Input: 只有 [SOS] 標記
        # Shape: [1, 1]
        dec_input_ids = torch.tensor([SOS_TOKEN], dtype=torch.long, device=device).unsqueeze(0) 
        
        # 3. 轉換動作向量為 Tensor，以便在 CUDA 上索引
        directions_tensor = torch.tensor(directions_vectors, dtype=torch.float32, device=device)

        # 4. 自迴歸循環：預測 T-1 個動作，直到軌跡長度達到 T
        # 循環次數 T_length - 1 (例如 99 次)
        for t_step in range(T_length - 1):
            
            # --- A. 準備輸入和遮罩 ---
            
            # Encoder Input: 傳遞當前已生成的所有座標 (P1 到 P_{t})
            # Shape: [1, t, 2]
            encoder_input = trajectory_coords.clone().to(device)
            
            # Decoder Mask: 根據當前 dec_input_ids 的長度生成因果遮罩
            src_att = None
            trg_att = subsequent_mask(dec_input_ids.shape[1]).to(device)
            
            # --- B. 模型預測 ---
            
            # 獲取 Softmax 機率分佈 (Prediction)
            # 輸出 Shape: [1, 1, 9]
            output_probabilities = model.predict(
                encoder_input, 
                dec_input_ids, 
                src_att, 
                trg_att
            )[:, -1, :] # 只取最後一個時間步的預測結果
            
            # --- C. 選擇動作 (Greedy Decoding) ---
            
            # 選擇機率最高的動作 ID (Argmax)
            next_action_id = output_probabilities.argmax(dim=-1).item() 
            
            # --- D. 更新序列與座標 ---
            
            # 1. 擴展 Decoder Input 序列 (將新 ID 加入歷史，用於下一步預測)
            predicted_action_tensor = torch.tensor([next_action_id], dtype=torch.long, device=device).unsqueeze(0)
            dec_input_ids = torch.cat((dec_input_ids, predicted_action_tensor), dim=1)
            
            # 2. 查找位移向量 (ID -> Delta_x, Delta_y)
            displacement_vector = directions_tensor[next_action_id].unsqueeze(0).unsqueeze(0) 
            
            # 3. 計算新座標 P_t = P_{t-1} + Delta_x
            last_point = trajectory_coords[:, -1:, :]
            new_point = last_point + displacement_vector
            
            # 4. 更新完整軌跡
            trajectory_coords = torch.cat((trajectory_coords, new_point), dim=1)
            
    # 返回 [T, 2] 格式的軌跡，並移除批次維度
    return trajectory_coords.squeeze(0).detach()


def transformer_batch_predict_logits(
    model: nn.Module, 
    history_coords: torch.Tensor, 
    SOS_TOKEN: int, 
    device: torch.device
) -> torch.Tensor:
    
    G, T, _ = history_coords.shape

    safe_history_coords = history_coords.clone()
    inf_mask = torch.isinf(safe_history_coords)
    safe_history_coords[inf_mask] = 0.0

    dec_input_ids = torch.tensor([SOS_TOKEN], dtype=torch.long, device=device).repeat(G, 1) # Shape [G, 1]
    
    src_mask = create_src_mask_from_coords(history_coords)

    trg_mask = subsequent_mask(dec_input_ids.shape[1]).to(device)
    
    out_logits = model(
        safe_history_coords, 
        dec_input_ids, 
        src_mask, 
        trg_mask
    ) # shape: (G, 1, 9)

    logits_G_9 = out_logits.squeeze(1)
    log_probs_G_9 = F.log_softmax(logits_G_9, dim=-1)

    return log_probs_G_9

def create_src_mask_from_coords(coords: torch.Tensor) -> torch.Tensor:

    is_padding = torch.isinf(coords[:, :, 0])

    return is_padding.unsqueeze(1)