# core/models/csi_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    """
    Residual Block specifically for CSI Feature Extraction.
    Maintains the 'topology' of the signal by preserving gradient flow.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to handle dimension changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # The magic of ResNet: f(x) + x
        out = F.relu(out)
        return out


class CSIEncoder(nn.Module):
    """
    Topology-Preserving CSI Encoder.
    
    Architecture Philosophy:
    1. Stem: Expands physics-features into high-dim latent space.
    2. Backbone: ResNet-1D stages to capture frequency-selective fading patterns.
    3. Neck: Global Pooling to ensure robustness to small jitters.
    4. Head: Projection to the specific manifold dimension (Channel Charting).

    Input:  (B, In_Channels, M_Subcarriers)
    Output: (B, Embedding_Dim)
    """
    def __init__(self, in_channels: int, embedding_dim: int = 16):
        super().__init__()

        # Initial Feature Expansion
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Deep Feature Extraction (The "Brain")
        # Captures multipath patterns at different scales
        self.layer1 = ResBlock1D(64, 64, stride=1)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, 256, stride=2)
        
        # The Manifold Projection Head
        # Projects high-dim features onto the low-dim topological map
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
            nn.Flatten(),             # (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim) # The final coordinate z
        )

        # Initialization (Crucial for Contrastive Learning)
        self._initialize_weights()

    def forward(self, x):
        # x: (B, C, M)
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        z = self.head(x)
        
        # Normalize to unit sphere (Hypersphere Manifold)
        # This is standard for Triplet Loss / Contrastive Learning
        z = F.normalize(z, p=2, dim=1)
        return z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)