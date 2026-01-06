# modules/factory.py

import torch
from strategies.analysis_strategy import BaselineAnalysisStrategy, ProposedAnalysisStrategy
from strategies.location_strategy import BaselineLocationStrategy, ProposedLocationStrategy

class SystemFactory:
    """
    Static Factory: Creates strategy objects based on the system configuration.
    """
    
    @staticmethod
    def create_signal_processor(config, device):
        """
        Creates an ISignalProcessor instance based on SYSTEM_MODE.
        """
        mode = config.get('SYSTEM_MODE', 'BASELINE').upper()
        
        if mode == 'BASELINE':
            return BaselineAnalysisStrategy(config, device)
        
        elif mode == 'PROPOSED':
            return ProposedAnalysisStrategy(config, device)
            
        else:
            raise ValueError(f"[Factory] Unknown SYSTEM_MODE: {mode}")

    @staticmethod
    def create_location_estimator(config, device, reference_grid, directions_vectors, transformer_model=None):
        """
        Creates an ILocationEstimator instance based on SYSTEM_MODE.
        """
        mode = config.get('SYSTEM_MODE', 'BASELINE').upper()
        
        if mode == 'BASELINE':
            return BaselineLocationStrategy(
                config, 
                device, 
                reference_grid, 
                directions_vectors
            )
        
        elif mode == 'PROPOSED':
            return ProposedLocationStrategy(
                config, 
                device, 
                reference_grid, 
                directions_vectors,
                transformer_model
            )
            
        else:
            raise ValueError(f"[Factory] Unknown SYSTEM_MODE: {mode}")