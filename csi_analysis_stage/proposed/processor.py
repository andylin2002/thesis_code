import numpy as np
from typing import Dict, Any, Optional
import os
import torch

# Import common utilities
from .._common.preprocessing import run_data_processor
from .._common.extraction.mmp import MMP_Algorithm

class ProposedProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        One-time initialization.
        """
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.mmp_engine = MMP_Algorithm(config=config)
        
        # Cache config parameters
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']

    def process(self, raw_csi_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process batch data.
        """
        # --- Data Preprocessing ---
        processed_csi = run_data_processor(
            raw_csi_data=raw_csi_data,
            config=self.config
        )

        # Prepare Batch
        num_batch = self.num_ap * self.num_sample
        # (QT, N, M) - Ensure reshape logic aligns with data structure
        batch_input_csi = processed_csi.reshape(num_batch, *processed_csi.shape[2:]).contiguous()

        # --- Feature Extraction ---
        aoa_all_paths, tof_all_paths, gain_all_paths = self.mmp_engine.estimate_aoa_tof_batch(
            input_csi=batch_input_csi
        )

        # AoA & ToF
        aoa_main_flat = aoa_all_paths[:, 0]
        tof_main_flat = tof_all_paths[:, 0]

        # AoA_Spread & ToF_Spread
        aoa_sprd_flat = self._calculate_rms_spread(aoa_all_paths, gain_all_paths)
        tof_sprd_flat = self._calculate_rms_spread(tof_all_paths, gain_all_paths)

        # Stacking & Reshape
        # Feature combo: [aoa_main, aoa_sprd, tof_main, tof_sprd]
        features_stacked_flat = torch.stack([
            aoa_main_flat, 
            aoa_sprd_flat, 
            tof_main_flat, 
            tof_sprd_flat
        ], dim=1)

        features = features_stacked_flat.reshape(self.num_ap, self.num_sample, 4)

        DEBUG = True
        if DEBUG:
            self._save_debug_info(features)

        return features
    
    def _calculate_rms_spread(self, values: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        """
        Calculate RMS Spread (Energy-weighted standard deviation).
        Args:
            values: (Batch, L) - AoA or ToF
            gains: (Batch, L) - Path Amplitudes
        Returns:
            spread: (Batch,)
        """
        # Convert amplitude to power
        power = gains.pow(2)
        
        # Calculate total power (avoid div by zero)
        total_power = torch.sum(power, dim=1, keepdim=True)
        total_power = torch.clamp(total_power, min=1e-9)

        # Weighted Mean
        weighted_mean = torch.sum(power * values, dim=1, keepdim=True) / total_power

        # Weighted Variance
        variance = torch.sum(power * (values - weighted_mean).pow(2), dim=1, keepdim=True) / total_power

        # Root Mean Square
        return torch.sqrt(variance).squeeze(1)

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