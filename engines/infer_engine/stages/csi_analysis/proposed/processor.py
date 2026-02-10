# csi_analysis_stage/proposed/processor.py

import numpy as np
from typing import Dict, Any, Optional
import os
import torch

# Import common utilities
from .._common.preprocessing.aggregation import run_csi_aggregation
from .._common.extraction.mmp import MMPAlgorithm

class ProposedProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        One-time initialization.
        """
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.mmp_engine = MMPAlgorithm(config=config)
        
        # Cache config parameters
        self.ap_data = config.get('ACCESS_POINTS', {})
        self.num_ap = len(self.ap_data)
        self.num_sample = config['NUM_SAMPLE']

    def process(self, raw_csi_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process batch data.
        """
        # --- Data Preprocessing ---
        processed_csi = run_csi_aggregation(
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
        C = aoa_all_paths.shape[1]
        aoa_cand_flat = aoa_all_paths[:, :C]
        tof_cand_flat = tof_all_paths[:, :C]
        gain_cand_flat = gain_all_paths[:, :C]

        # AoA_Spread & ToF_Spread
        aoa_sprd_flat = self._calculate_aoa_spread(aoa_all_paths, gain_all_paths).unsqueeze(1).expand(-1, C)
        tof_sprd_flat = self._calculate_tof_spread(tof_all_paths, gain_all_paths).unsqueeze(1).expand(-1, C)

        # Stacking & Reshape
        # Feature combo: [aoa_cand, aoa_sprd, tof_cand, tof_sprd, gain_cand]
        features_stacked_flat = torch.stack([   # [batch, C, 5]
            aoa_cand_flat, 
            aoa_sprd_flat, 
            tof_cand_flat, 
            tof_sprd_flat, 
            gain_cand_flat
        ], dim=2)

        # [Q, T, C, 5]
        features = features_stacked_flat.reshape(self.num_ap, self.num_sample, C, 5)

        DEBUG = True
        if DEBUG:
            self._save_debug_info(features)

        return features
    
    def _calculate_aoa_spread(self, aoa_deg: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        """
        Calculate Angular Spread using Circular Statistics.
        Input: aoa_deg (Batch, L) in Degrees
        Output: spread (Batch,) in Degrees
        """
        # Calculate weights
        power = gains.pow(2)
        total_power = torch.sum(power, dim=1, keepdim=True)
        total_power = torch.clamp(total_power, min=1e-9)
        weights = power / total_power

        # Turn angle to cos, sin
        aoa_rad = torch.deg2rad(aoa_deg)
        
        # R_x, R_y is the weighted average vector component
        R_x = torch.sum(weights * torch.cos(aoa_rad), dim=1, keepdim=True)
        R_y = torch.sum(weights * torch.sin(aoa_rad), dim=1, keepdim=True)
        
        # 3. Calculate the Mean Resultant Length (R)
        # R values closer to 1 indicate higher angular concentration (low spread).
        # R values closer to 0 indicate higher angular dispersion (high spread).
        R_len = torch.sqrt(R_x.pow(2) + R_y.pow(2))
        R_len = torch.clamp(R_len, max=1.0)

        # 4. Calculate Circular Variance
        circ_var = 1.0 - R_len
        
        # Turn radius to degree
        spread_rad = torch.sqrt(2.0 * circ_var)
        spread_deg = torch.rad2deg(spread_rad).squeeze(1)

        # Avoid Spread = 0
        spread_deg = torch.clamp(spread_deg, min=1.0)
        
        return spread_deg
    
    def _calculate_tof_spread(self, tof_sec: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        """
        Calculate Linear Spread for Time-of-Flight.
        Input: tof_sec (Batch, L) in Seconds
        Output: spread (Batch,) in Seconds
        """
        # Calculate weights
        power = gains.pow(2)
        total_power = torch.sum(power, dim=1, keepdim=True)
        total_power = torch.clamp(total_power, min=1e-9)
        weights = power / total_power

        # Calculate Linear Weighted Mean
        weighted_mean = torch.sum(weights * tof_sec, dim=1, keepdim=True)

        # Calculate Weighted Variance
        variance = torch.sum(weights * (tof_sec - weighted_mean).pow(2), dim=1, keepdim=True)
        
        # Calculate standard deviation
        spread_sec = torch.sqrt(variance).squeeze(1)

        # Avoid Spread = 0
        spread_sec = torch.clamp(spread_sec, min=1e-9)

        return spread_sec

    def _save_debug_info(self, features):
        """Helper function to save debug info."""
        try:
            features_np = features.detach().cpu().numpy()
            save_path = "output/csi_features.npy"
            os.makedirs("output", exist_ok=True) # Ensure directory exists
            np.save(save_path, features_np)
            #print(f"[ProposedProcessor] Feature matrix saved to: {os.path.abspath(save_path)}")
            #print(f"[ProposedProcessor] Shape: {features_np.shape} (Expect: Q x T x C x 5)")
        except Exception as e:
            print(f"[Error] Failed to save feature matrix: {e}")