# engines/adapt_engine/stages/load/stage.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch

class LoadStage:
    """
    LoadStage acts as a buffer. 
    It accumulates single CSI packets until a full batch is ready.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device  # Note: Data usually stays on CPU until batch is full
        self.batch_size = int(config.get("ADAPT_BATCH_SIZE", 16))
        
        # Buffer to hold individual packets
        self.buffer: List[torch.Tensor] = []

    def accumulate(self, raw_csi: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Push a packet into the buffer.
        
        Args:
            raw_csi: Single packet tensor (Q, T, N, M)
            
        Returns:
            torch.Tensor (B, Q, T, N, M) if batch is full.
            None if buffering.
        """
        self.buffer.append(raw_csi)

        if len(self.buffer) >= self.batch_size:
            # Stack into a single batch tensor
            # Expected shape: (B, Q, T, N, M)
            batch_tensor = torch.stack(self.buffer, dim=0)
            
            # Clear buffer for next round
            self.buffer = []
            
            return batch_tensor

        return None