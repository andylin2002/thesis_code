import numpy as np

# (TODO)
class EM_Algorithm:
    def __init__(self, initial_params, feature_matrix):
        # 儲存初始參數和觀測數據
        self.params = initial_params 
        self.observations = feature_matrix 
        self.max_iterations = 20 # 預設最大迭代次數

    def _compute_emission_log_likelihood(self) -> np.ndarray:

        pass
        
    def _e_step(self):
        # ... 呼叫 construct_emission_log_likelihood.py 裡面的邏輯
        # ... 計算 Forward/Backward 概率 (Find Better Trajectory)
        pass
        
    def _m_step(self):
        # ... 更新 Pi, A, B 參數 (Find Better Model)
        pass
        
    def _check_convergence(self):
        # 檢查模型參數或對數似然值是否收斂
        return False
        
    def run_em_iterations(self) -> np.ndarray:
        """Runs the EM loop until convergence or max iterations is reached."""
        
        # 這裡會重複執行 E-Step 和 M-Step
        for i in range(self.max_iterations):
            self._e_step()
            self._m_step()
            if self._check_convergence():
                break
        
        # 最終輸出從 EM 狀態轉換回實際座標的序列
        # 這裡需要一個函式將最終的狀態機率轉換為 (T, 2) 座標
        # Placeholder: 輸出全零座標作為演示
        T_points = self.observations.shape[0]
        return np.zeros((T_points, 2), dtype=np.float32)