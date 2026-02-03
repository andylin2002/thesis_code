# csi_analysis_stage/_common/extraction/delay_estimator.py

import torch

def estimate_delay_batch(
        num_batch: torch.Tensor, 
        batch_input_csi: torch.Tensor, 
    ):

    magnitude_csi = batch_input_csi.abs()
    variance = torch.var(magnitude_csi.reshape(num_batch, -1), dim=1)
    delay_flat = 10 * torch.log10(variance + 1e-9)

    return delay_flat