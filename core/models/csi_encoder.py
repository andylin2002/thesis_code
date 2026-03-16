# core/models/csi_encoder.py

import torch
import torch.nn as nn

class CSIEncoder(nn.Module):
    """
    CSIEncoder: Extracts spatial-spectral features and preserves the temporal dimension (T).
    Input:  [Batch*Q, T, N-1, M, 2]
    Output: [Batch*Q, T, latent_dim]
    """
    def __init__(
        self,
        num_antennas: int,
        num_subcarriers: int,
        cnn_channels: list = [16, 32],
        cnn_kernel: tuple = (3, 3),
        gru_hidden: int = 64,
        gru_layers: int = 1,
        mlp_hidden: int = 32,
        latent_dim: int = 128,
        proj_dim: int = 64
    ):
        super().__init__()

        # 1. Spatial-Spectral CNN
        cnn_layers = []
        in_ch = 2  # Amplitude and phase difference
        for out_ch in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=cnn_kernel, padding=1))
            cnn_layers.append(nn.BatchNorm2d(out_ch))
            cnn_layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate flattened feature dimension
        n_spatial = num_antennas - 1
        m_spectral = num_subcarriers
        self.feature_dim = cnn_channels[-1] * n_spatial * m_spectral

        # 2. Temporal GRU
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True
        )

        # 3. Latent MLP
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, latent_dim)
        )

        # 4. Projection Head (Optional for contrastive tasks)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        B_flat, T, N, M, C = x.shape
        
        # Extract spatial features for each time step independently
        x_cnn = x.permute(0, 1, 4, 2, 3).contiguous()  # [B_flat, T, C, N, M]
        x_cnn = x_cnn.view(B_flat * T, C, N, M)
        
        cnn_out = self.cnn(x_cnn)
        cnn_out = cnn_out.view(B_flat, T, -1)  # Restore temporal dimension T
        
        # Process sequence temporally (keep all T steps)
        gru_out, _ = self.gru(cnn_out)  # gru_out: [B_flat, T, gru_hidden]
        
        # Map to latent space (Linear layer applies automatically to the last dimension)
        z_q = self.mlp(gru_out)  # z_q: [B_flat, T, latent_dim]
        
        if return_projection:
            return self.projection(z_q)  # [B_flat, T, proj_dim]
            
        return z_q