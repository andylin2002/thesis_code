import torch
from core.interfaces import ISignalProcessor

from csi_analysis_stage.baseline.processor import run_csi_analysis

class BaselineAnalysisStrategy(ISignalProcessor):
    """
    Strategy for Baseline signal processing.
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def extract(self, raw_csi_block):
        """
        Input: [Q, T, N, M]
        Output: Features [Q, T, 3] (Power, Angle, Delay)
        """
        feature_matrix = run_csi_analysis(
            raw_csi_data=raw_csi_block,
            config=self.config
        )
        
        return {
            'mode': 'BASELINE',
            'features': feature_matrix
        }

class ProposedAnalysisStrategy(ISignalProcessor):
    """
    Strategy for Proposed signal processing (MMP + SPD).
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def extract(self, raw_csi_block):
        """
        Input: [Q, T, N, M]
        Output: Features [Q, T, 4], SPD [Q, T, N, N]
        """
        # 1. Extract MMP features (AoA, ToF, AS, DS) -> [Q, T, 4]
        feature_matrix = run_csi_analysis(
            raw_csi_data=raw_csi_block,
            config=self.config
        )
        
        # 2. Compute SPD Matrix -> [Q, T, N, N]
        spd = self._compute_spd(raw_csi_block)
        
        return {
            'mode': 'PROPOSED',
            'features': feature_matrix,
            'spd': spd
        }

    def _compute_spd(self, raw_csi):
        """
        Computes Spatial Covariance Matrix R = H * H^H.
        Input: [Q, T, N, M] (APs, Time, Antennas, Subcarriers)
        Output: [Q, T, N, N]
        """
        # PyTorch matmul operates on the last two dims.
        # We want (N, M) @ (M, N) -> (N, N).
        
        # 1. Conjugate Transpose of the last two dims: (N, M) -> (M, N)
        if raw_csi.is_complex():
            raw_csi_H = raw_csi.conj().transpose(-1, -2)
        else:
            raw_csi_H = raw_csi.transpose(-1, -2)
            
        # 2. Matrix Multiplication: H * H^H
        # [..., N, M] @ [..., M, N] = [..., N, N]
        R = torch.matmul(raw_csi, raw_csi_H)
        
        # 3. Normalize by number of subcarriers (M)
        M = raw_csi.shape[-1]
        R = R / M
        
        return R