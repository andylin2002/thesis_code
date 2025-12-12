import torch
from typing import Dict, Any
from core.interfaces import ILocationEstimator
from indoor_location_stage._common import grid_tools

# (For Baseline)
from indoor_location_stage.baseline.estimator import BaselineEstimator

# (For Proposed)
from indoor_location_stage.proposed.estimator import ProposedEstimator

class BaselineLocationStrategy(ILocationEstimator):
    """
    Strategy for Baseline localization (Hard EM / Viterbi).
    Directly wraps the legacy function.
    """
    def __init__(self, config, device, reference_grid, directions_vectors):
        self.config = config
        self.device = device
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors

        # Pre-calculate AP info ONCE during initialization
        self.num_ap = len(config['ACCESS_POINTS'])
        self.ap_data_info = {
            'locations': grid_tools.get_ap_locations(config, self.num_ap, device),
            'orientations': torch.tensor(
                [data.get('ORIENTATION_DEG', 0) for data in config['ACCESS_POINTS'].values()], 
                dtype=torch.float32, 
                device=device
            )
        }

    def estimate(self, signal_data: Dict[str, Any]) -> torch.Tensor:
        # Features shape: [Q, T, 3] (Power, Angle, Delay)
        features = signal_data['features']

        # Initialize EM algorithm in 'MARKOV' mode
        estimator = BaselineEstimator(
            features=features,
            config=self.config,
            reference_grid=self.reference_grid,
            ap_data=self.ap_data_info, 
            device=self.device
        )
        
        # Solve (Run EM + Viterbi)
        trajectory = estimator.solve()
            
        return trajectory

class ProposedLocationStrategy(ILocationEstimator):
    """
    Strategy for Proposed localization (Physics-Aware AI).
    Handles Buffer slicing and EPD injection.
    """
    def __init__(self, config, device, reference_grid, directions_vectors, transformer_model=None):
        self.config = config
        self.device = device
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors
        self.transformer_model = transformer_model

        # Pre-calculate AP info ONCE during initialization
        self.num_ap = len(config['ACCESS_POINTS'])
        self.ap_data_info = {
            'locations': grid_tools.get_ap_locations(config, self.num_ap, device),
            'orientations': torch.tensor(
                [data.get('ORIENTATION_DEG', 0) for data in config['ACCESS_POINTS'].values()], 
                dtype=torch.float32, 
                device=device
            )
        }

    def estimate(self, signal_data: Dict[str, Any]) -> torch.Tensor:
        """
        Executes the Physics-Aware pipeline
        """
        # Extract Data
        features = signal_data['features']
        buffer = signal_data.get('buffer')
        spd = signal_data.get('spd') 
        
        # Instantiate the Proposed Estimator
        estimator = ProposedEstimator(
            features=features,
            buffer=buffer,
            spd=spd,
            config=self.config,
            reference_grid=self.reference_grid,
            ap_data=self.ap_data_info,
            device=self.device,
            model=self.transformer_model
        )

        # Solve (Physics Layer -> AI Layer -> Viterbi Fusion)
        trajectory = estimator.solve()

        # Extract EPD for Training Worker
        if hasattr(estimator, 'epd') and estimator.epd is not None:
            signal_data['epd'] = estimator.epd.detach().cpu()
            
        return trajectory