# engines/infer_engine/stages/signal_processing/_common/extraction/music.py

import math
import torch
from typing import Dict, Any

class MUSICAlgorithm:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the 1D MUSIC Algorithm for dominant AoA extraction.
        """
        LIGHT_SPEED = 299792458.0

        self.device = config.get('device', torch.device('cpu'))
        
        # 1. Read hardware parameters from config
        csi_dims = config.get('CSI_DIMENSIONS', {})
        self.num_antennas = csi_dims.get('NUM_RX_ANTENNAS', 3)
        self.d = config.get('ANTENNA_DISTANCE', 0.03)
        
        # 2. Calculate wavelength (lambda = c / f)
        freq = config.get('CARRIER_FREQUENCY_HZ', 5180000000.0)
        c = LIGHT_SPEED
        self.lam = c / freq

        # 3. Setup grid search parameters
        self.angle_res = config.get('ANGLE_RES', 1.0)
        self.angles = torch.arange(-90, 90 + self.angle_res, self.angle_res, device=self.device)
        angles_rad = torch.deg2rad(self.angles)
        
        # 4. Pre-compute steering vectors matrix A
        # Shape of A: (num_angles, num_antennas)
        idx = torch.arange(self.num_antennas, device=self.device)
        phase = -1j * 2 * math.pi * (self.d / self.lam) * torch.sin(angles_rad).unsqueeze(1) * idx
        
        self.A = torch.exp(phase) 
        self.A_conj = self.A.conj()

    def estimate_aoa_batch(self, batch_input_csi: torch.Tensor) -> torch.Tensor:
        """
        Extract the dominant AoA for a batch of CSI data.
        
        Args:
            batch_input_csi: Tensor of shape (Batch, N_t, M)
            
        Returns:
            peaks: Tensor of shape (Batch,) with estimated AoA in degrees.
        """
        B, N, M = batch_input_csi.shape
        
        # 1. Calculate sample covariance matrix R = (1/M) * H * H^H
        # Shape of R: (B, N_t, N_t)
        R = (1.0 / M) * torch.bmm(batch_input_csi, batch_input_csi.conj().transpose(1, 2))
        
        # 2. Eigen decomposition (L is ascending, V contains eigenvectors)
        L, V = torch.linalg.eigh(R)
        
        # 3. Construct noise subspace (exclude the largest eigenvector)
        # Shape of U_n: (B, N_t, N_t-1)
        U_n = V[:, :, :-1]
        
        # Calculate U_n * U_n^H
        # Shape of Un_UnH: (B, N_t, N_t)
        Un_UnH = torch.bmm(U_n, U_n.conj().transpose(1, 2))
        
        # 4. Calculate pseudo-spectrum using optimized einsum
        # Equation: a^H * Un * Un^H * a 
        # Result shape: (Batch, num_angles)
        denom = torch.einsum('ai, bij, aj -> ba', self.A_conj, Un_UnH, self.A)
        
        # Calculate P(φ) = 1 / |a^H * Un * Un^H * a|
        spectrum = 1.0 / (torch.abs(denom) + 1e-12)
        
        # 5. Find the peak angle for each item in the batch
        best_idx = torch.argmax(spectrum, dim=1)
        peaks = self.angles[best_idx]
        
        return peaks