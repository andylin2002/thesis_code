# indoor_location_stage/location_estimator.py
import torch

# TODO

class LocationEstimator:
    """
    [空殼] 為了讓 Baseline 模式能順利 Import 而存在的暫時檔案。
    Proposed 模式的核心邏輯將會填寫在這裡。
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        print("[PhysicsLayer] Initialized (Skeleton).")

    def compute_epd(self, features, spd=None):
        # 暫時回傳假資料，反正 Baseline 不會呼叫這裡
        print("[PhysicsLayer] compute_epd called (Placeholder)")
        return torch.zeros(1, 1, 1024) 

    def run_viterbi_step(self, epd, ai_transition, grid, vectors):
        # 暫時回傳 None
        return None