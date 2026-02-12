# engines/infer_engine/strategies/processing_strategy.py

from core.interfaces import IProcessor

from ..stages.signal_processing.baseline.processor import BaselineProcessor
from ..stages.signal_processing.proposed.processor import ProposedProcessor

class BaselineProcessorStrategy(IProcessor):
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
        # Extract features (power, angle, delay spread)
        features = self.processor.process(
            raw_csi_data=raw_csi_block
        )
        
        return features

class ProposedProcessorStrategy(IProcessor):
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
        Output: Features [Q, T, C, 5]
        """
        # Extract MMP features (aoa_cand, aoa_sprd, tof_cand, tof_sprd, gain_cand)
        features = self.processor.process(
            raw_csi_data=raw_csi_block
        )
        
        return features