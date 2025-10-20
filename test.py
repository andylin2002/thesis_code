import numpy as np

# === 1. 讀取原始 CSI 檔 ===
input_path = "csi_sample.npy"   # 你的原始 .npy 檔案
output_path = "csi_sample_TQNM.npy"  # 轉換後輸出的檔案名稱

# 載入 .npy 檔案
csi = np.load(input_path)
print(f"原始形狀: {csi.shape}, 資料型態: {csi.dtype}")

# === 2. 檢查形狀是否為 (Q, N, M, T) ===
if csi.ndim != 4:
    raise ValueError(f"資料維度應該是4維，但目前是 {csi.ndim} 維")

# === 3. 調整維度順序 (Q, N, M, T) → (T, Q, N, M) ===
# np.transpose 會根據索引重新排列維度
# 原順序 index: Q(0), N(1), M(2), T(3)
# 目標順序 index: T(3), Q(0), N(1), M(2)
csi_reordered = np.transpose(csi, (3, 0, 1, 2))

print(f"轉換後形狀: {csi_reordered.shape}")

# === 4. 儲存轉換後的結果 ===
np.save(output_path, csi_reordered)
print(f"已儲存轉換後的檔案至: {output_path}")

# === 5. 驗證重新載入是否一致 ===
csi_check = np.load(output_path)
print(f"驗證載入形狀: {csi_check.shape}, dtype: {csi_check.dtype}")
