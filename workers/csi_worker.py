import torch
import numpy as np
from collections import deque
from core.interfaces import BaseWorker
from modules.factory import SystemFactory
from queue import Empty

from transformer.transformer_tool import create_transformer_instance

class CSI_Worker(BaseWorker):
    """
    Worker responsible for real-time signal processing and location estimation.
    Operates in two modes (Baseline/Proposed) determined by SystemFactory.
    """
    
    def __init__(self, name, config, queues, stop_event, reference_grid, directions_vectors):
        # Initialize common attributes via BaseWorker
        super().__init__(name, config, queues, stop_event)
        
        # Store specialized attributes required for location strategies
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors
        
        # Placeholders for strategies and models
        self.signal_processor = None
        self.location_estimator = None
        self.history_buffer = None

    def _setup(self):
        """
        Setup strategies and buffers before the main loop starts.
        """
        # 1. Create Strategies via Factory
        # The factory handles the logic of 'BASELINE' vs 'PROPOSED' internally
        self.signal_processor = SystemFactory.create_signal_processor(
            self.config, 
            self.device
        )
        
        self.location_estimator = SystemFactory.create_location_estimator(
            self.config,
            self.device,
            self.reference_grid,
            self.directions_vectors,
            transformer_model=None # Initially None, updated via model_queue
        )
        
        # 2. Initialize Sliding Window Buffer (Proposed Mode Only)
        buffer_len = self.config.get('BUFFER_LEN', 10)
        self.history_buffer = deque(maxlen=buffer_len)
        
        print(f"[{self.name}] Setup complete. Strategies loaded.")

    def _loop(self):
        """
        Main inference loop.
        """
        in_queue = self.queues['data']   # Raw CSI input
        out_queue = self.queues['result'] # Output to AI Worker / UI
        model_queue = self.queues['model']    # Receive updated AI model

        while not self.stop_event.is_set():
            try:
                raw_csi_block = in_queue.get(timeout=0.1) 
            except Empty:
                continue
            
            try:
                # Move to GPU for Processing
                raw_csi_block = raw_csi_block.to(self.device)

                # Check for Model Updates (Hot-Swapping)
                if not model_queue.empty():
                    new_model_config = model_queue.get()
                    self._update_model(new_model_config)

                # Strategy Execution: Signal Processing
                # Returns dict with 'mode', 'features', ('spd')
                processed_data = self.signal_processor.extract(raw_csi_block)

                if 'features' in processed_data:
                        self.history_buffer.append(processed_data['features'])
                        processed_data['buffer'] = list(self.history_buffer)

                # Strategy Execution: Location Estimation
                # Returns Tensor/np.array of the predicted path
                trajectory = self.location_estimator.estimate(processed_data)

                # Output Handling
                if processed_data.get('mode') == 'PROPOSED':
                    # For Proposed mode, we send the "Input + Label" package to AI Worker
                    training_pkg = {
                        'input': processed_data, # Features, SPD, EPD
                        'pseudo_gt': trajectory  # Path from Viterbi
                    }
                    training_pkg_cpu = self._recursive_detach_cpu(training_pkg)
                    out_queue.put(training_pkg_cpu)
                    print(f"[{self.name}] Result SENT to queue.")
                else:
                    # For Baseline, just output the path (or handle differently)
                    # Here we wrap it to match the queue expectation if needed
                    trajectory_cpu = self._recursive_detach_cpu(trajectory)
                    out_queue.put(trajectory_cpu)
                    print(f"[{self.name}] Result SENT to queue.")

                in_queue.task_done()

            except Exception as e:
                print(f"[{self.name}] Error in loop processing: {e}")
                import traceback
                traceback.print_exc()
                in_queue.task_done()

    def _update_model(self, model_config):
        """
        Handles both 'Hot-Swap' and 'Cold-Start' scenarios.
        """
        version = model_config.get('version', 'Unknown')
        print(f"[{self.name}] Received AI Model Update (V{version})")
        
        # Validation: Ensure strategy supports AI
        if not hasattr(self.location_estimator, 'transformer_model'):
            print(f"[{self.name}] Warning: Current strategy ignores AI model.")
            return

        try:
            weights = model_config['weights']
            
            # Case 1: Model exists -> Hot swap weights
            if self.location_estimator.transformer_model is not None:
                self.location_estimator.transformer_model.load_state_dict(weights)
                
            # Case 2: Cold Start -> Instantiate, load, and assign
            else:
                # Create model instance
                n_directions = 9 
                model = create_transformer_instance(self.config, n_directions, self.device)
                
                # Load weights and set to eval mode
                model.load_state_dict(weights)
                model.eval() 
                
                # Inject model into the strategy
                self.location_estimator.transformer_model = model
                
            print(f"[{self.name}] Model V{version} updated successfully.")
            
        except Exception as e:
            print(f"[{self.name}] Failed to update model: {e}")

    def _recursive_detach_cpu(self, data):
        """
        Helper: Recursively move Tensors in dict/list/tuple to CPU.
        Removes GPU dependency for IPC safety.
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        elif isinstance(data, dict):
            return {k: self._recursive_detach_cpu(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._recursive_detach_cpu(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._recursive_detach_cpu(v) for v in data)
        else:
            return data