import numpy as np
from typing import Dict, Any, Optional

from .data_processor import run_data_processor
from .feature_extractor import run_feature_extractor

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_csi_analysis(
        raw_csi_data: torch.Tensor, 
        config: Dict[str, Any]
) -> Optional[torch.Tensor]:

##### --- Data Preprocessing (on GPU) ---

    processed_csi = run_data_processor(
        raw_csi_data=raw_csi_data,
        config=config
    )

##### --- Feature Extraction (on GPU) ---

    feature_matrix = run_feature_extractor(
        processed_csi=processed_csi,
        config=config
    )

    # (FIXME: DEBUG)=======================================
    DEBUG = True
    if DEBUG:
        if feature_matrix is not None:
            print("\n[DEBUG] Feature Matrix (F) Check:")
            print(f"  Shape: {feature_matrix.shape}")

            # 特徵維度分析 (假設 feature_matrix 的形狀是 [Q, T, 3])
            # F[:, :, 0] -> Power
            # F[:, :, 1] -> Angle
            # F[:, :, 2] -> Delay
            
            # 1. 計算 Power (索引 0) 的統計量
            print("\n  --- Power Features (索引 0) ---")
            power_features = feature_matrix[:, :, 0]
            print(power_features[0, :])
            print(f"  Max/Min: {torch.max(power_features):.4f} / {torch.min(power_features):.4f}")
            print(f"  Mean: {torch.mean(power_features):.6f}")
            print(f"  Std Dev: {torch.std(power_features):.6f}")

            # 2. 計算 Angle (索引 1) 的統計量
            print("\n  --- Angle Features (索引 1) ---")
            angle_features = feature_matrix[:, :, 1]
            print(angle_features[0, :])
            print(f"  Max/Min: {torch.max(angle_features):.4f} / {torch.min(angle_features):.4f}")
            print(f"  Mean: {torch.mean(angle_features):.6f}")
            print(f"  Std Dev: {torch.std(angle_features):.6f}")
            
            # 3. 計算 Delay (索引 2) 的統計量
            print("\n  --- Delay Features (索引 2) ---")
            delay_features = feature_matrix[:, :, 2]
            print(delay_features[0, :])
            print(f"  Max/Min: {torch.max(delay_features):.4f} / {torch.min(delay_features):.4f}")
            print(f"  Mean: {torch.mean(delay_features):.6f}")
            print(f"  Std Dev: {torch.std(delay_features):.6f}")

            # 總體檢查 (保持總體檢查，用於快速確認)
            print("\n  --- Overall Check ---")
            std_total = torch.std(feature_matrix)
            if std_total < 1e-6:
                print("🚨🚨 WARNING: 特徵矩陣 F 接近常數！")
            else:
                print(f"  Total Std Dev (All Features): {std_total:.6f}")

        else:
            print("[DEBUG] Feature Matrix is None. CSI Analysis failed.")
    # (FIXME: DEBUG)=======================================

    return feature_matrix
    