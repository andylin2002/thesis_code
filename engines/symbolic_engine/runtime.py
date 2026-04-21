# engines/symbolic_engine/runtime.py

import torch
from queue import Empty

from .modules.factory import SystemFactory


class SymbolicRuntime:
    """
    
    """

    def __init__(self, config, device, reference_grid, directions_vectors, model_queue=None):
        self.config = config
        self.device = device
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors
        self.model_queue = model_queue

        # Strategies / models
        self.signal_processor = None
        self.location_estimator = None

        self._setup()

    def _setup(self):
        # Create Strategies via Factory (baseline/proposed handled inside)
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

    def step(self, raw_csi_block_cpu):
        """
        Run one inference step for one CSI block.
        """
        # Move to GPU for processing
        raw_csi_block = raw_csi_block_cpu.to(self.device)

        # Apply latest gating update (state_dict only)
        if self.model_queue is not None:
            self._apply_latest_gating_state_dict()

        # Stage 1: Signal Processing
        features = self.signal_processor.extract(raw_csi_block)
        
        aggregated_csi = getattr(self.signal_processor, "aggregated_csi", None)

        # Stage 2: Location Estimation
        trajectory = self.location_estimator.estimate(features, aggregated_csi)

        emission_log_probs_qgt = getattr(self.location_estimator, "emission_log_probs_qgt", None)
        posterior_gt = getattr(self.location_estimator, "posterior_gt", None)
        reliability = getattr(self.location_estimator, "reliability", None)

        # Package (future-proof: dict payload)
        pkg = {
            "trajectory": trajectory,
            "aggregated_csi": aggregated_csi, 
            "emission_log_probs_qgt": emission_log_probs_qgt,
            "posterior_gt": posterior_gt, 
            "reliability": reliability, 
            "features": features, # DEBUG
            "aggregated_csi": aggregated_csi # DEBUG
        }

        # Convert to CPU-safe package
        pkg_cpu = self._recursive_detach_cpu(pkg)
        return pkg_cpu

    def _recursive_detach_cpu(self, data):
        """
        
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

    def _apply_latest_gating_state_dict(self):
        """
        
        """
        latest = None
        while True:
            try:
                latest = self.model_queue.get_nowait()
            except Empty:
                break

        if latest is None:
            return

        state_dict = latest.get("state_dict", None)

        # Forward to estimator (only Proposed estimator should implement this)
        if hasattr(self.location_estimator, "set_gating_state_dict"):
            self.location_estimator.set_gating_state_dict(state_dict)
