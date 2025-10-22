# csi_analysis_stage/feature_extractor/mmp_core.py

import torch
from typing import Dict, Any, Tuple
import math

import numpy as np

#(TODO) complete mmp algorithm
class MMP_Algorithm:
    def __init__(self, config: Dict[str, Any]):
        
    ##### --- Parameter Setup ---        
        self.config = config
        self.d_antenna_spacing = config['ANTENNA_DISTANCE']
        self.fc_carrier_frequency_hz = config['CARRIER_FREQUENCY_HZ']             
        self.B_channel_bandwith_hz = config['CHANNEL_BANDWIDTH_HZ']
        self.N_antenna = config['CSI_DIMENSIONS']['NUM_RX_ANTENNAS']
        self.M_subcarrier = config['CSI_DIMENSIONS']['NUM_SUBCARRIERS']
        
        self.alpha = math.floor((self.M_subcarrier + 1) / 2)
        self.beta = self.M_subcarrier - self.alpha + 1

        c = 299792458.0  # Speed of light (m/s)
        self.wavelength = c / self.fc_carrier_frequency_hz

    def estimate_aoa_tof_batch(
        self, 
        input_csi: torch.Tensor, # (QT, N, M)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
    ##### --- Construct CSI Enhance Matrix ---
        
        input_csi_enhance = self._construct_enhance_matrix(input_csi)

    ##### --- Find partition the Unitary Matrix U of the left singular values ---

        Us_left_singular_vectors, L_multipath = self._get_left_singular_vectors(input_csi_enhance)

    ##### --- ToF ---
        tof_tensor, eigenvector_matrix = self._estimate_tof_logic(Us_left_singular_vectors, L_multipath) 
        
    ##### --- AoA ---
        aoa_tensor_flat = self._estimate_aoa_logic(Us_left_singular_vectors, L_multipath, eigenvector_matrix)
        
        return aoa_tensor_flat, tof_tensor
    

    def _estimate_tof_logic(self, Us_left_singular_vectors: torch.Tensor, L_multipath: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # (TODO)
        

        return 0

    def _estimate_aoa_logic(self, Us_left_singular_vectors: torch.Tensor, L_multipath: torch.Tensor, eigenvector_matrix: torch.Tensor) -> torch.Tensor: # (TODO)
        

        return 0

    def _construct_enhance_matrix(self, matrix: torch.Tensor) -> torch.Tensor:

        alpha = self.alpha
        beta = self.beta
        
        C1 = self._construct_hankel_matrix(matrix[:, 0, :], alpha, beta)
        C2 = self._construct_hankel_matrix(matrix[:, 1, :], alpha, beta)
        C3 = self._construct_hankel_matrix(matrix[:, 2, :], alpha, beta)

        row1 = torch.cat([C1, C2], dim=2)
        row2 = torch.cat([C2, C3], dim=2)
        enhance_matrix = torch.cat([row1, row2], dim=1)

        return enhance_matrix
    
    def _construct_hankel_matrix(self, vector: torch.Tensor, row: int, col: int) -> torch.Tensor:

        # 'unfold' function needs the unsqueezing dimension
        vector_unfoldable = vector.unsqueeze(1) # (QT, 1, M)
        hankel_unfolded = vector_unfoldable.unfold(2, col, 1) # (QT, 1, row, col)
        hankel_matrix = hankel_unfolded.squeeze(1).contiguous() # (QT, row, col)

        return hankel_matrix

    def _get_left_singular_vectors(self, matrix: torch.Tensor, threshold_ratio: float=0.99) -> Tuple[torch.Tensor, torch.Tensor]:

        batch, row, col = matrix.shape

        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        S_squared = S.pow(2)
        total_energy = S_squared.sum(dim=1, keepdim=True)
        cumulative_energy = torch.cumsum(S_squared, dim=1)
        normalized_cumulative_ratio = cumulative_energy / total_energy

        # shape: (batch, L(t,q))
        L_mask = (normalized_cumulative_ratio >= threshold_ratio)
        L_rank_indices = L_mask.int().argmax(dim=1) + 1
        L_rank_indices = torch.clamp(L_rank_indices, min=1, max = col)

        L_rank_float = L_rank_indices.float()
        L_mean = L_rank_float.mean()
        L_std = L_rank_float.std()

        L_multipath_float = L_mean + 1.0 * L_std
        L_multipath = math.ceil(L_multipath_float)
        L_multipath = int(torch.clamp(torch.tensor(L_multipath), min=1, max=col).item())

        Us = U[:, :, :L_multipath]

        return Us, L_multipath
    
    def _permutation(self, even_row_matrix: torch.Tensor) -> torch.Tensor:

        pass