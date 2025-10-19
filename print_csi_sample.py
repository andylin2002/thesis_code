import scipy.io
import numpy as np
import os # 用於處理檔案路徑

# --- 設定檔案路徑 ---
base_dir = "/home/mcs/dadalin/dataset/CSI-dataset-for-indoor-localization/Lab Dataset"
real_file_path = os.path.join(base_dir, "coordinate 1-100/coordinate101.mat")
imag_file_path = os.path.join(base_dir, "imaginary_part/imaginary101.mat")

# 假設資料鍵（Key）是 'myData'
data_key = 'myData'
output_sample_file = 'csi_sample.npy'

loaded_csi = np.load(output_sample_file)

# 驗證資料是否正確：形狀與類型應與儲存時一致
print(f"載入數據形狀: {loaded_csi.shape}")
print(f"載入數據類型: {loaded_csi.dtype}")

print("\n--- 數據內容範例 (複數) ---")
# print(loaded_csi)
