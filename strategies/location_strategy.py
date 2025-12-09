import torch
from core.interfaces import ILocationEstimator

# Legacy entry point (For Baseline)
from indoor_location_stage import run_indoor_location

# New components (For Proposed)
from indoor_location_stage.location_estimator import LocationEstimator as PhysicsLayer

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

    def estimate(self, signal_data):
        # Features shape: [Q, T, 3] (Power, Angle, Delay)
        feature_matrix = signal_data['features']
        
        # Call legacy function with MARKOV mode
        trajectory = run_indoor_location(
            feature_matrix=feature_matrix,
            config=self.config,
            reference_grid=self.reference_grid,
            model=None,
            mode='MARKOV',
            directions_vectors=self.directions_vectors
        )
            
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
        
        # Initialize Physics Layer (Soft EM & EPD Calculator)
        self.physics_layer = PhysicsLayer(config, device)
        
        # Initialize Viterbi/Path Manager (Assuming it's available or wrapped in PhysicsLayer)
        # For now, we assume physics_layer has a method to run viterbi, 
        # or you can import your Viterbi class here.
        # self.viterbi = ViterbiPathManager(...) 

    def estimate(self, signal_data):
        """
        Executes the Physics-Aware pipeline:
        1. Compute EPD (0.5s)
        2. AI Inference (10s history -> 0.5s intent)
        3. Viterbi Fusion (0.5s)
        """
        # 1. Get current 0.5s features
        current_features = signal_data['features'] 
        
        # 2. Compute EPD (Physics Layer) - Only for current 0.5s
        # Using the SPD from signal_data if needed for Dynamic Variance
        spd = signal_data.get('spd')
        epd = self.physics_layer.compute_epd(current_features, spd)
        
        # [CRITICAL STEP] Inject EPD back into dictionary
        # This allows the Worker to send it to the Training Worker later
        signal_data['epd'] = epd.detach().cpu() 

        # 3. AI Inference (Transformer) - Uses 10s Buffer
        ai_transition = None
        buffer_list = signal_data.get('buffer')
        
        if self.transformer_model is not None and buffer_list:
            # (A) Stack Buffer: List of Tensors -> (1, Time=10s, Dim)
            input_seq = torch.cat(buffer_list, dim=1) 
            
            with torch.no_grad():
                # (B) Full sequence inference
                full_transition = self.transformer_model(input_seq)
                
                # (C) Slice: Take only the part corresponding to current 0.5s
                T_curr = current_features.shape[1]
                ai_transition = full_transition[:, -T_curr:, :]

        # 4. Viterbi Fusion - Only for current 0.5s
        # Note: You need to ensure you have a Viterbi implementation accessible here.
        # Example call:
        trajectory = self.physics_layer.run_viterbi_step(
            epd, 
            ai_transition, 
            self.reference_grid,
            self.directions_vectors
        )
            
        return trajectory