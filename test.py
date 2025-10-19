# test.py

import numpy as np
import os

# --- 檔案路徑設定 ---
CSI_FILE_PATH = 'csi_sample.npy' 
OUTPUT_FILE_PATH = '4_ap_simulated_csi.npy'

def create_multi_ap_data(csi_path: str, num_aps: int = 4):
    """
    載入單一 AP 的 CSI 數據，並將其複製成多 AP (4) 的模擬數據。
    
    Args:
        csi_path (str): 原始 CSI 數據 (.npy) 路徑。
        num_aps (int): 模擬的 AP 數量 (預期為 4)。
        
    Returns:
        np.ndarray: 複製後的數據，形狀為 (4, 3, 30, 1500)。
    """
    try:
        # 1. 載入原始 CSI 數據
        # 原始形狀: (3, 30, 1500)
        single_ap_csi = np.load(csi_path)
        
        # 驗證原始形狀 (確保是我們預期的格式)
        if single_ap_csi.shape != (3, 30, 1500):
            print(f"Error: Loaded CSI shape {single_ap_csi.shape} is not the expected (3, 30, 1500).")
            return None
        
        # 2. 使用 np.tile 複製數據
        # np.tile(A, reps): 沿著每個維度重複 A 的次數
        # 我們希望在新的第 0 維度重複 4 次，其他維度重複 1 次。
        # 為了使用 np.tile，需要先使用 np.newaxis (或 None) 創建一個新的維度 (shape: 1, 3, 30, 1500)
        
        temp_reshaped = single_ap_csi[np.newaxis, :, :, :] # Shape (1, 3, 30, 1500)
        
        # 沿著第 0 維度重複 4 次
        multi_ap_csi = np.tile(temp_reshaped, (num_aps, 1, 1, 1))
        
        # 3. 驗證最終形狀
        print(f"\nOriginal CSI shape: {single_ap_csi.shape}")
        print(f"Final Multi-AP CSI shape: {multi_ap_csi.shape}")
        
        if multi_ap_csi.shape == (num_aps, 3, 30, 1500):
            print("Creation successful: Data is ready for 4 APs.")
            return multi_ap_csi
        else:
            return None
            
    except FileNotFoundError:
        print(f"Error: File not found at {csi_path}. Please check the path.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == '__main__':
    # 執行數據創建
    multi_ap_data = create_multi_ap_data(CSI_FILE_PATH, num_aps=4)
    
    if multi_ap_data is not None:
        # 儲存為新的 .npy 檔案以供後續使用 (可選)
        np.save(OUTPUT_FILE_PATH, multi_ap_data)
        print(f"\nData successfully saved to {OUTPUT_FILE_PATH}")