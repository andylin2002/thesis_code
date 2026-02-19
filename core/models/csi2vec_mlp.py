# core/models/csi2vec_mlp.py

import torch
import torch.nn as nn

class CSI2VecMLP(nn.Module):
    """
    Embedding function g_phi as described in CSI2Vec paper.
    MLP structure: {D, 32, D'} where D' is embedding dimension.
    """
    def __init__(self, input_dim: int, embedding_dim: int = 16):
        super().__init__()
        
        # Architecture: {D, 32, D'} activations per layer 
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True), # ReLU for all except last layer [cite: 223]
            nn.Linear(32, embedding_dim)
        )
        
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input features are flattened vectors [cite: 137]
        return self.model(x)

    def _init_weights(self):
        # Glorot (Xavier) initialization [cite: 224]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)