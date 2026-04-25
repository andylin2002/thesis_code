# engines/symbolic_engine/modules/factory.py

from ..strategies.processing_strategy import BaselineProcessorStrategy, ProposedProcessorStrategy
from ..strategies.estimation_strategy import BaselineEstimatorStrategy, ProposedEstimatorStrategy

class SystemFactory:
    """
    Static Factory: Creates strategy objects based on the system configuration.
    """
    
    @staticmethod
    def create_signal_processor(config, device):
        """
        Creates an ISignalProcessor instance based on METHOD.
        """
        method = config.get('METHOD', 'BASELINE').upper()
        
        if method == 'BASELINE':
            return BaselineProcessorStrategy(config, device)
        
        elif method == 'PROPOSED':
            return ProposedProcessorStrategy(config, device)
            
        else:
            raise ValueError(f"[Factory] Unknown METHOD: {method}")

    @staticmethod
    def create_location_estimator(config, device, reference_grid, directions_vectors):
        """
        Creates an ILocationEstimator instance based on METHOD.
        """
        method = config.get('METHOD', 'BASELINE').upper()
        
        if method == 'BASELINE':
            return BaselineEstimatorStrategy(
                config, 
                device, 
                reference_grid, 
                directions_vectors
            )
        
        elif method == 'PROPOSED':
            return ProposedEstimatorStrategy(
                config, 
                device, 
                reference_grid, 
                directions_vectors
            )
            
        else:
            raise ValueError(f"[Factory] Unknown METHOD: {method}")