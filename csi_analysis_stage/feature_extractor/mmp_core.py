# csi_analysis_stage/feature_extractor/mmp_core.py

import torch
from typing import Dict, Any, Tuple, List
import math

#(TODO) complete mmp algorithm
class MMP_Algorithm:
    def __init__(self, config: Dict[str, Any]):
        
    ##### --- Parameter Setup ---        
        self.config = config
        self.d_antenna_spacing = config['ANTENNAS_DISTENCE']
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
    ##### --- Construct CSI Enhance Matrix ---
        
        input_csi_enhance = self._construct_enhance_matrix(input_csi)

    ##### --- Find partition the Unitary Matrix U of the left singular values ---

        Us_left_singular_vectors, L_multipath = self._get_left_singular_vectors(input_csi_enhance)

    ##### --- ToF ---
        tof_tensor_list, eigenvector_matrix = self._estimate_tof_logic(Us_left_singular_vectors, L_multipath) 
        
    ##### --- AoA ---
        aoa_tensor_flat = self._estimate_aoa_logic(Us_left_singular_vectors, L_multipath, eigenvector_matrix) 

        print(f"      [MMP Core] AoA Output Shape: {aoa_tensor_flat.shape}")
        print(f"      [MMP Core] ToF Output Type: List[Tensor] (Length {len(tof_tensor_list)})")
        
        return aoa_tensor_flat, tof_tensor_list

    def _estimate_aoa_logic(self, csi_snapshot: torch.Tensor) -> torch.Tensor:
        """Placeholder for the actual AoA estimation logic (e.g., MUSIC/ESPRIT)."""
        # 為了演示，假設輸出是單一角度值 (對應一個 batch entry)
        # 這裡的計算會用到 self.wavelength
        # ... 複雜的 AoA 計算邏輯 ...
        return torch.rand(csi_snapshot.shape[0], device=csi_snapshot.device) * 360 

    def _estimate_tof_logic(self, csi_snapshot: torch.Tensor) -> List[torch.Tensor]:
        """Placeholder for ToF estimation logic, returning a list of tensors."""
        # 模擬計算出不同長度的 ToF 集合
        N_batches = csi_snapshot.shape[0]
        tof_tensor_list: List[torch.Tensor] = []
        
        for i in range(N_batches):
            L_i = torch.randint(low=5, high=15, size=(1,)).item()
            tof_set = torch.rand(L_i, device=csi_snapshot.device) * 1e-7
            tof_tensor_list.append(tof_set)
        
        # 返回長度不一的 ToF 集合列表，用於後續的 Delay Spread (tau) 計算
        return tof_tensor_list

    # --- 公開批次處理方法 (供 Coordinator 呼叫) ---

    def _construct_enhance_matrix(self, matrix: torch.Tensor) -> torch.Tensor:

        B_batch, N_antenna, M_subcarrier = matrix.shape
        alpha = self.alpha
        beta = self.beta

        hankel_list: List[torch.Tensor] = []
        
        C1 = self._construct_hankel_matrix(matrix[:, 0, :], alpha, beta)
        C2 = self._construct_hankel_matrix(matrix[:, 1, :], alpha, beta)
        C3 = self._construct_hankel_matrix(matrix[:, 2, :], alpha, beta)

        enhance_matrix = torch.block([
            [C1, C2],
            [C2, C3]
        ])

        return enhance_matrix
    
    def _construct_hankel_matrix(self, vector: torch.Tensor, row: int, col: int) -> torch.Tensor:

        # 'unfold' function needs the dimension
        vector_unfoldable = vector.unsqueeze(1) # (QT, 1, M)
        hankel_unfolded = vector_unfoldable.unfold(2, row, 1) # (QT, 1, row, col)
        hankel_matrix = hankel_unfolded.squeeze(1).contiguous() # (QT, row, col)

        return hankel_matrix

    def _get_left_singular_vectors(self, matrix: torch.Tensor, threshold_ratio: float=0.99):

        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        S_squared = S.pow(2)
        total_energy = S_squared.sum(dim=1, keepdim=True)
        cumulative_energy = torch.cumsum(S_squared, dim=1)
        normalized_cumulative_ratio = cumulative_energy / total_energy

        L_mask = (normalized_cumulative_ratio >= threshold_ratio)
        L_rank_indices = L_mask.int().argmax(dim=1)
        L_max = L_rank_indices.max().item() + 1
        L_rank = max(1, L_max)

        Us_left_singular_vectors = U[:, :, :L_max]

        return Us_left_singular_vectors, L_rank
    