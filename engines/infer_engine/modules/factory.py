# engines/infer_engine/modules/factory.py

from ..strategies.processing_strategy import BaselineProcessorStrategy, ProposedProcessorStrategy
from ..strategies.estimation_strategy import BaselineEstimatorStrategy, ProposedEstimatorStrategy

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
            return BaselineProcessorStrategy(config, device)
        
        elif mode == 'PROPOSED':
            return ProposedProcessorStrategy(config, device)
            
        else:
            raise ValueError(f"[Factory] Unknown SYSTEM_MODE: {mode}")

    @staticmethod
    def create_location_estimator(config, device, reference_grid, directions_vectors):
        """
        Creates an ILocationEstimator instance based on SYSTEM_MODE.
        """
        mode = config.get('SYSTEM_MODE', 'BASELINE').upper()
        
        if mode == 'BASELINE':
            return BaselineEstimatorStrategy(
                config, 
                device, 
                reference_grid, 
                directions_vectors
            )
        
        elif mode == 'PROPOSED':
            return ProposedEstimatorStrategy(
                config, 
                device, 
                reference_grid, 
                directions_vectors
            )
            
        else:
            raise ValueError(f"[Factory] Unknown SYSTEM_MODE: {mode}")