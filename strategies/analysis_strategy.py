import torch
from core.interfaces import ISignalProcessor

from csi_analysis_stage.baseline.processor import BaselineProcessor
from csi_analysis_stage.proposed.processor import ProposedProcessor

class BaselineAnalysisStrategy(ISignalProcessor):
    """
    Strategy for Baseline signal processing.
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.processor = BaselineProcessor(config)

    def extract(self, raw_csi_block):
        """
        Input: [Q, T, N, M]
        Output: Features [Q, T, 3] (Power, Angle, Delay)
        """
        features = self.processor.process(
            raw_csi_data=raw_csi_block
        )
        
        return {
            'mode': 'BASELINE',
            'features': features
        }

class ProposedAnalysisStrategy(ISignalProcessor):
    """
    Strategy for Proposed signal processing (MMP + SPD).
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.processor = ProposedProcessor(config)

    def extract(self, raw_csi_block):
        """
        Input: [Q, T, N, M]
        Output: Features [Q, T, 4], SPD [Q, T, N, N]
        """
        # 1. Extract MMP features (AoA, ToF, AS, DS) -> [Q, T, 4]
        features = self.processor.process(
            raw_csi_data=raw_csi_block
        )
        
        # 2. Compute SNR -> [Q, T, N, N]
        snr = self._compute_snr_from_covariance(raw_csi_block)
        
        return {
            'mode': 'PROPOSED',
            'features': features,
            'snr': snr
        }

    def _compute_snr_from_covariance(self, raw_csi: torch.Tensor) -> torch.Tensor:
        """
        Computes SNR via Eigenvalue Decomposition (EVD).
        SNR = Max_Eigenvalue (Signal) / Min_Eigenvalue (Noise).
        
        Input: [Q, T, N, M]
        Output: [Q, T] (dB)
        """
        # --- 1. Compute Spatial Covariance Matrix R ---
        
        # Conjugate Transpose: (N, M) -> (M, N)
        if raw_csi.is_complex():
            raw_csi_H = raw_csi.conj().transpose(-1, -2)
        else:
            raw_csi_H = raw_csi.transpose(-1, -2)
            
        # Matrix Mult: [..., N, M] @ [..., M, N] = [..., N, N]
        R = torch.matmul(raw_csi, raw_csi_H)
        
        # Normalize by num subcarriers (M)
        M = raw_csi.shape[-1]
        R = R / M

        # --- 2. Eigenvalue Decomposition ---
        
        # Use eigvalsh for Hermitian matrices (faster/stable)
        # Returns eigenvalues in ascending order
        eigvals = torch.linalg.eigvalsh(R)
        
        # --- 3. Calculate SNR ---
        
        # Max Eigenvalue = Signal Power + Noise Power
        signal_power = eigvals[..., -1] 
        
        # Min Eigenvalue = Noise Floor
        # Add epsilon to avoid division by zero
        noise_power = eigvals[..., 0] + 1e-9
        
        # Linear SNR
        snr_linear = signal_power / noise_power
        
        # Convert to dB
        snr_db = 10 * torch.log10(snr_linear)
        
        return snr_db