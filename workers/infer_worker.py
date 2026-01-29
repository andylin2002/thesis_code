# workers/infer_worker.py

import torch
from queue import Empty

from core.interfaces import BaseWorker
from modules.factory import SystemFactory

class INFER_Worker(BaseWorker):
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
        )
        
        print(f"[{self.name}] Setup complete. Strategies loaded.")

    def _loop(self):
        """
        Main inference loop.
        """
        in_queue = self.queues['data']                  # Raw CSI input
        out_queue = self.queues['result']               # Output to AI Worker / UI
        debug_queue = self.queues.get("debug", None)     # To Main Process (Saving/Plotting)
        model_queue = self.queues.get('model', None)    # Receive updated AI model

        while not self.stop_event.is_set():
            try:
                raw_csi_block = in_queue.get(timeout=0.1) 
            except Empty:
                continue
            
            try:
                # Move to GPU for Processing
                raw_csi_block = raw_csi_block.to(self.device)

                # Apply latest gating update if exists (state_dict only)
                if model_queue is not None:
                    self._apply_latest_gating_state_dict(model_queue)

                # Strategy Execution: Signal Processing
                features = self.signal_processor.extract(raw_csi_block)

                # Strategy Execution: Location Estimation
                trajectory = self.location_estimator.estimate(features)

                pkg = {
                    "trajectory": trajectory,
                    "training": {
                        "features": features,
                    },
                }
                pkg_cpu = self._recursive_detach_cpu(pkg)
                out_queue.put(pkg_cpu)

                # Debug queue (optional)
                if debug_queue is not None:
                    debug_queue.put(pkg_cpu["trajectory"])

                in_queue.task_done()

            except Exception as e:
                print(f"[{self.name}] Error in loop processing: {e}")
                import traceback
                traceback.print_exc()
                in_queue.task_done()

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
        
    def _apply_latest_gating_state_dict(self, model_queue):
        """
        Drain model_queue and apply the latest state_dict to the location strategy.
        Payload format is expected to be:
            {"state_dict": <dict or None>}
        """
        latest = None
        while not model_queue.empty():
            latest = model_queue.get()

        if latest is None:
            return

        # Strict: state_dict only (no model instance here)
        state_dict = latest.get("state_dict", None)

        # Forward to strategy (only Proposed strategy should implement this)
        if hasattr(self.location_estimator, "set_gating_state_dict"):
            self.location_estimator.set_gating_state_dict(state_dict)
