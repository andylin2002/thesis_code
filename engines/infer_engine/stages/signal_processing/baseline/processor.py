# engines/infer_engine/stages/signal_processing/baseline/processor.py

import numpy as np
from typing import Dict, Any, Optional
import os
import torch

# Import common utilities
from .._common.preprocessing.aggregation import run_csi_aggregation
from .._common.extraction.music import MUSICAlgorithm
from .._common.extraction.mmp import MMPAlgorithm
from .._common.extraction import power_extractor, delay_estimator

class BaselineProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        One-time initialization.
        """
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.aoa_method = config.get('BASELINE_AOA_METHOD', 'music').lower()

        if self.aoa_method == 'music':
            self.extractor_engine = MUSICAlgorithm(config=config)
        elif self.aoa_method == 'mmp':
            self.extractor_engine = MMPAlgorithm(config=config)
        else:
            raise ValueError(
                f"{self.aoa_method} is the Unknown AoA Extraction Method."
            )
        
        # Cache config parameters
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']

    def process(self, raw_csi_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process batch data.
        """
        # Data Preprocessing
        processed_csi = run_csi_aggregation(
            raw_csi_data=raw_csi_data,
            config=self.config
        )

        # Prepare Batch
        num_batch = self.num_ap * self.num_sample
        # (QT, N, M) - Ensure reshape logic aligns with data structure
        batch_input_csi = processed_csi.reshape(num_batch, *processed_csi.shape[2:]).contiguous()

        # Feature Extraction
        # --- Power ---
        power_flat = power_extractor.extract_power_batch(
            input_csi=batch_input_csi
        )

        # --- Angle ---
        if self.aoa_method == 'mmp':
            aoa_all_paths, tof_all_paths, gain_all_paths = self.extractor_engine.estimate_aoa_tof_batch(
                input_csi=batch_input_csi
            )
            angle_flat = aoa_all_paths[:, 0]
        else:
            # self.aoa_method == 'music'
            angle_flat = self.extractor_engine.estimate_aoa_batch(
                batch_input_csi=batch_input_csi
            )

        # --- Delay ---
        delay_flat = delay_estimator.estimate_delay_batch(num_batch, batch_input_csi)

        # Stacking & Reshape
        # Feature combo: [Power, Angle, Delay]
        features_stacked_flat = torch.stack([
            power_flat, 
            angle_flat, 
            delay_flat
        ], dim=1)

        features = features_stacked_flat.reshape(self.num_ap, self.num_sample, 3)

        DEBUG = True
        if DEBUG:
            self._save_debug_info(features)

        return features

    def _save_debug_info(self, features):
        """Helper function to save debug info."""
        try:
            features_np = features.detach().cpu().numpy()
            save_path = "output/csi_features.npy"
            os.makedirs("output", exist_ok=True) # Ensure directory exists
            np.save(save_path, features_np)
            print(f"[BaselineProcessor] Feature matrix saved to: {os.path.abspath(save_path)}")
            print(f"[BaselineProcessor] Shape: {features_np.shape} (Expect: Q x T x 3)")
        except Exception as e:
            print(f"[Error] Failed to save feature matrix: {e}")