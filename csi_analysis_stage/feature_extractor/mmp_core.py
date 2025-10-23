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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
    ##### --- Construct CSI Enhance Matrix ---
        
        input_csi_enhance = self._construct_enhance_matrix(input_csi)

    ##### --- Find partition the Unitary Matrix U of the left singular values ---

        Us_left_singular_vectors = self._get_left_singular_vectors(input_csi_enhance)

    ##### --- ToF ---
        tof_tensor, eigv_y, principal_left_singular_vector = self._estimate_tof_logic(Us_left_singular_vectors) 
        
    ##### --- AoA ---
        aoa_tensor_flat, eigv_x = self._estimate_aoa_logic(Us_left_singular_vectors, principal_left_singular_vector)
        
        return aoa_tensor_flat, tof_tensor, eigv_x, eigv_y
    

    def _estimate_tof_logic(self, Us_left_singular_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
    ##### --- Prepare the element that Lemma will use ---
        permutation_Us = self._permutation(Us_left_singular_vectors)
        hat_Us_upper = permutation_Us[:, :-2, :]
        hat_Us_lower = permutation_Us[:, 2:, :]

    ##### --- Find Diagonalizable Matrix ---
        pinv_hat_Us_upper = torch.linalg.pinv(hat_Us_upper)
        target_matrix = pinv_hat_Us_upper @ hat_Us_lower

    ##### --- Diagonalization ---
        eigv_y, eigenvector_matrix = torch.linalg.eig(target_matrix)
        # eigv_y.shape: (batch, L)
        # eigenvector_matrix.shape: (batch, L, L)

    ##### --- We Can Use 'principal_left_singular_vector' to Find eigv_x1 ---
        principal_left_singular_vector = eigenvector_matrix[:, :, 0]
        # principal_left_singular_vector.shape = (batch, L)

    ##### --- eigv_y -> ToF ---
        tof_tensor = self._eigv_y_to_tof(eigv_y)

        return tof_tensor, eigv_y, principal_left_singular_vector


    def _estimate_aoa_logic(self, Us_left_singular_vectors: torch.Tensor, principal_left_singular_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        alpha = self.alpha

    ##### --- Find Diagonalizable Matrix ---
        Us_upper = Us_left_singular_vectors[:, :alpha, :]
        Us_lower = Us_left_singular_vectors[:, alpha:, :]

        pinv_Us_upper = torch.linalg.pinv(Us_upper)
        target_matrix = pinv_Us_upper @ Us_lower

    ##### --- Diagonalization ---
        eigv_x, _ = torch.linalg.eig(target_matrix)
        # eigv_x.shape: (batch, L)

    ##### --- Find eigv_x1 ---
        principal_left_singular_vector = principal_left_singular_vector.unsqueeze(-1)

        A = (Us_upper @ principal_left_singular_vector).mH
        B = Us_lower @ principal_left_singular_vector
        C = Us_upper @ principal_left_singular_vector

        eigv_x1 = ((A @ B) / (A @ C)).squeeze()

    ##### --- eigv_x1 -> AoA ---
        aoa_tensor_flat = self._eigv_x1_to_aoa(eigv_x1)

        return aoa_tensor_flat, eigv_x


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

    def _get_left_singular_vectors(self, matrix: torch.Tensor, threshold_ratio: float=0.99) -> torch.Tensor:

        batch, row, col = matrix.shape

    ##### --- Ignore Paths with no Influence by Energy Threshold ---
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        S_squared = S.pow(2)
        total_energy = S_squared.sum(dim=1, keepdim=True)
        cumulative_energy = torch.cumsum(S_squared, dim=1)
        normalized_cumulative_ratio = cumulative_energy / total_energy

    ##### --- Choose Numbers of path ---
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

    ##### --- Get Us ---
        Us = U[:, :, :L_multipath]

        return Us
    
    def _permutation(self, even_row_matrix: torch.Tensor) -> torch.Tensor:

        batch, even_row, col = even_row_matrix.shape
        if even_row % 2 != 0:
             raise ValueError("Can't Permute the matrix since it's row is not even!")
        row = even_row // 2

        matrix_B_2_R_C = even_row_matrix.reshape(batch, 2, row, col)

        matrix_B_R_2_C = matrix_B_2_R_C.transpose(1, 2)

        permuted_even_row_matrix = matrix_B_R_2_C.reshape(batch, even_row, col)

        return permuted_even_row_matrix
    
    def _eigv_y_to_tof(self, eigv_y: torch.Tensor) -> torch.Tensor:

        ln_y = torch.log(eigv_y)

        numerator = 1j * self.M_subcarrier * ln_y
        denominator = 2 * torch.pi * self.B_channel_bandwith_hz

        tof_tensor = (numerator / denominator).real

        return tof_tensor

    def _eigv_x1_to_aoa(self, eigv_x1: torch.Tensor) -> torch.Tensor:

        ln_x1 = torch.log(eigv_x1)

        numerator = 1j * self.wavelength * ln_x1
        denominator = 2 * torch.pi * self.d_antenna_spacing

        sin_phi_term = (numerator / denominator).real
        aoa_radians = torch.arcsin(sin_phi_term)
        aoa_degrees = aoa_radians * (180.0 / torch.pi)

        return aoa_degrees