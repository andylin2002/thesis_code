import numpy as np
import scipy.io
import os
import sys

def generate_fixed_directions(file_path: str = "directions.mat", grid_step: float = 1.0):
    """
    生成 3x3 九宮格的位移向量，並儲存為 'directions.mat'。
    """
    
    # 定義 3x3 九宮格的相對位移 (dx, dy)，步長為 grid_step
    # 按照從 (-1,-1) 到 (1,1) 的順序排列 (這裡使用標準的九宮格順序)
    movements = np.array([
        # ID 0 到 ID 8
        [-1.0, -1.0],  # ID 0: Down-Left (西南)
        [ 0.0, -1.0],  # ID 1: Down (南)
        [ 1.0, -1.0],  # ID 2: Down-Right (東南)
        
        [-1.0,  0.0],  # ID 3: Left (西)
        [ 0.0,  0.0],  # ID 4: Stop (中心)
        [ 1.0,  0.0],  # ID 5: Right (東)
        
        [-1.0,  1.0],  # ID 6: Up-Left (西北)
        [ 0.0,  1.0],  # ID 7: Up (北)
        [ 1.0,  1.0]    # ID 8: Up-Right (東北)
    ], dtype=np.float32) * grid_step

    N_DIRECTIONS = movements.shape[0] 
    
    if N_DIRECTIONS != 9:
        print(f"錯誤：預期 9 個方向，但找到 {N_DIRECTIONS}。", file=sys.stderr)
        sys.exit(1)

    # 儲存到指定路徑，使用鍵名 'directions'
    scipy.io.savemat(file_path, {'directions': movements})

    return file_path

def print_directions_data(file_path: str = "directions.mat"):
    """載入指定的 directions.mat 檔案並打印內容。"""
    print("\n--- DIRECTIONS.MAT 檔案內容驗證 ---")
    
    if not os.path.exists(file_path):
        print(f"FATAL ERROR: The file '{file_path}' was not found after generation.")
        sys.exit(1)

    try:
        mat = scipy.io.loadmat(file_path)
        
        # 核心：使用新的鍵名 'directions' 提取數據
        if 'directions' not in mat:
            print(f"ERROR: '{file_path}' 中找不到預期的鍵 'directions'。請檢查生成腳本。")
            sys.exit(1)
            
        directions = mat['directions']
        N_DIRECTIONS = directions.shape[0]
        
        print(f"檔案載入成功，路徑: {os.path.abspath(file_path)}")
        print(f"總方向數量 (N_CLUSTERS): {N_DIRECTIONS}")
        
        print("\n九宮格動作向量內容 (dx, dy):")
        print("-----------------------------------")
        
        for i in range(N_DIRECTIONS):
            dx = directions[i, 0]
            dy = directions[i, 1]
            
            # 定義動作描述以增加清晰度
            description = ""
            if dx == 0 and dy == 0:
                description = "中心 (Stop)"
            elif dx == -1 and dy == -1:
                description = "左下 (South-West)"
            # ... 您可以添加其他描述 ...
            
            print(f"ID {i}: [dx={dx:>6.2f}, dy={dy:>6.2f}] ({description})")
            
        print("-----------------------------------")
        
    except Exception as e:
        print(f"處理檔案時發生錯誤：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    # 1. 設置參數
    OUTPUT_FILE_NAME = "directions.mat"
    
    # 2. 生成檔案
    generated_file = generate_fixed_directions(file_path=OUTPUT_FILE_NAME)
    
    # 3. 立即檢查和打印
    print_directions_data(generated_file)

    # 可選：在驗證後保留檔案供 main.py 使用
    # os.remove(generated_file)