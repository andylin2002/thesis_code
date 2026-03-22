# engines/symbolic_engine/stages/signal_processing/_common/extraction/delay_estimator.py

import torch

def estimate_delay_batch(batch_input_csi: torch.Tensor) -> torch.Tensor:
    B = batch_input_csi.shape[0]
    magnitude_csi = batch_input_csi.abs()
    variance = torch.var(magnitude_csi.reshape(B, -1), dim=1, unbiased=False)
    delay_flat = 10.0 * torch.log10(variance + 1e-9)
    return delay_flat