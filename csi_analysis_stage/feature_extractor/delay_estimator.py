# csi_analysis_stage/feature_extractor/delay_estimator.py

import torch
from typing import Tuple
import torch.nn.functional as F

def estimate_delay_batch(
        input_csi: torch.Tensor,
        tof_tensor: torch.Tensor, 
        eigv_x: torch.Tensor, 
        eigv_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    device = input_csi.device
    batch_size, N, M = input_csi.shape
    _, L = tof_tensor.shape

    n_idx = torch.arange(N, device=device).view(1, N, 1, 1)
    m_idx = torch.arange(M, device=device).view(1, 1, M, 1)

    x_vec = eigv_x.view(batch_size, 1, 1, L)
    y_vec = eigv_y.view(batch_size, 1, 1, L)

    Phi = torch.pow(x_vec, n_idx) * torch.pow(y_vec, m_idx) # (Batch, N, M, L)

    h = input_csi.view(batch_size, N * M, 1) # 攤平觀測值
    P = Phi.view(batch_size, N * M, L)

    kappa = torch.linalg.lstsq(P, h).solution.squeeze(-1) # (Batch, L)


    weights = torch.abs(kappa).pow(2)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
    mean_delay = (weights * tof_tensor).sum(dim=1, keepdim=True)

    diff_sq = (tof_tensor - mean_delay).pow(2)
    weighted_variance = (weights * diff_sq).sum(dim=1) # (Batch,)
    
    rms_spread = torch.sqrt(weighted_variance)
    delay_tensor_flat = 10 * torch.log10(rms_spread + 1e-9)

    return delay_tensor_flat