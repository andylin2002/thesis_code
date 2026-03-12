# core/models/csi_encoder.py

import torch.nn as nn

class CSIEncoder(nn.Module):
    """
    CSIEncoder for one AP.
    Input: [B, T, N-1, M, 2]  (preprocessed CSI: amplitude + phase difference)
    Output: latent vector z_q or projection vector for contrastive learning
    """

    def __init__(
        self,
        num_antennas=3,
        num_subcarriers=21,
        cnn_channels=[16, 32],
        cnn_kernel=(3, 3),
        gru_hidden=64,
        gru_layers=1,
        mlp_hidden=32,
        latent_dim=16,
        proj_dim=16
    ):
        super().__init__()

        # --- Spatial-Spectral CNN ---
        cnn_layers = []
        in_ch = 2  # real + imaginary / amplitude + phase
        for out_ch in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=cnn_kernel, padding=1))
            cnn_layers.append(nn.BatchNorm2d(out_ch))
            cnn_layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Flatten CNN output for GRU input
        n_spatial = num_antennas - 1
        m_spectral = num_subcarriers
        self.feature_dim = cnn_channels[-1] * n_spatial * m_spectral

        # --- Temporal GRU ---
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True
        )

        # --- Latent MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, latent_dim)
        )

        # --- Projection head for contrastive learning ---
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x, return_projection=False):
        """
        Forward pass for one AP.

        Args:
            x: [B, T, N-1, M, 2] preprocessed CSI
            return_projection: if True, return contrastive projection

        Returns:
            z_q: [B, latent_dim] latent vector
            or
            z_proj: [B, proj_dim] projection vector
        """
        B, T, N, M, C = x.shape
        # Merge time and batch for CNN
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B*T, C, N, M)
        x = self.cnn(x)
        x = x.view(B, T, -1)  # [B, T, feature_dim]

        # GRU over time dimension
        _, h_n = self.gru(x)
        h_n = h_n[-1]  # take last layer

        # Latent vector
        z_q = self.mlp(h_n)

        if return_projection:
            z_proj = self.projection(z_q)
            return z_proj

        return z_q