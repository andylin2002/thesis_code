# engines/neural_engine/stages/load.py

import torch
from typing import Any, Dict, List, Optional

class LoadStage:
    """Accumulates individual CSI packets into a full batch."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.batch_size = int(config.get("NEURAL_BATCH_SIZE", 16))
        self.buffer: List[torch.Tensor] = []

    def accumulate(self, raw_csi: torch.Tensor) -> Optional[torch.Tensor]:
        """Returns a stacked batch tensor if buffer is full, else None."""
        self.buffer.append(raw_csi)

        if len(self.buffer) >= self.batch_size:
            batch_tensor = torch.stack(self.buffer, dim=0)
            self.buffer = []
            return batch_tensor

        return None