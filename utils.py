# utils.py

import yaml
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import re
import json


def save_timing_report(timing_list, method, dataset_name, output_dir, wall_time_sec=None, warmup=5):
    """Summarize per-block timing dicts from SymbolicRuntime.step() into a JSON report."""
    usable = timing_list[warmup:] if len(timing_list) > warmup else timing_list
    if not usable:
        return None

    report = {
        "method": method,
        "dataset": dataset_name,
        "num_blocks_total": len(timing_list),
        "num_blocks_used": len(usable),
        "warmup_dropped": len(timing_list) - len(usable),
    }

    for k in usable[0].keys():
        vals = np.array([t[k] for t in usable], dtype=np.float64)
        report[k] = {
            "mean_ms": float(vals.mean()), "std_ms": float(vals.std()),
            "median_ms": float(np.median(vals)), "p95_ms": float(np.percentile(vals, 95)),
            "p99_ms": float(np.percentile(vals, 99)), "min_ms": float(vals.min()), "max_ms": float(vals.max()),
        }

    mean_total = report["total_ms"]["mean_ms"]
    report["compute_throughput_blocks_per_sec"] = 1000.0 / mean_total if mean_total > 0 else None

    if wall_time_sec:
        report["observed_wall_time_sec"] = float(wall_time_sec)
        report["observed_throughput_blocks_per_sec"] = len(timing_list) / wall_time_sec  # includes IPC/queue overhead

    path = os.path.join(output_dir, "timing_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Timing] {method}/{dataset_name}: mean={mean_total:.2f}ms p95={report['total_ms']['p95_ms']:.2f}ms "
          f"throughput={report['compute_throughput_blocks_per_sec']:.2f} blocks/s -> {path}")
    return report

def to_numpy(data):
    """Helper: Convert Tensor/List to Numpy safely."""
    if data is None: return None
    if hasattr(data, "detach"): return data.detach().cpu().numpy()
    return np.asarray(data)

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Loads configuration content from a YAML file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {file_path}")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error: Failed to parse YAML file: {exc}")
        return {}
    
def round_to_half(value: float) -> float:
    """Rounds a floating-point number to the nearest multiple of 0.5."""
    return round(value * 2) / 2

def generate_reference_grid(
    config: Dict[str, Any],
    x_bounds: list | None = None,
    y_bounds: list | None = None
) -> Tuple[torch.Tensor, list, list, int, int]:
    """
    Calculates the boundary of the localization area and generates
    a uniform grid of prediction points based on AP locations.
    """
    # 1. Extract AP coordinates from config
    ap_locations_list = []
    ap_data = config.get("ACCESS_POINTS", {})
    
    for ap_key in ap_data:
        location = ap_data[ap_key].get("LOCATION_M")
        if location and len(location) == 2:
            ap_locations_list.append(location)
            
    if not ap_locations_list:
        print("Warning: No valid AP coordinates found. Cannot generate grid.")
        return np.array([]), [0.0, 0.0], [0.0, 0.0]

    ap_locations_array = np.array(ap_locations_list)
    
    # 2. Extract grid parameters
    resolution = config.get("GRID_RESOLUTION_M")
    buffer = config.get("BUFFER_DISTANCE_M")

    # 3. Determine Boundary Limits (AP extremes)
    if x_bounds is not None and y_bounds is not None:
        # --- Use externally provided bounds ---
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

    else:
        # --- bounds from AP locations ---
        x_min_ap = np.min(ap_locations_array[:, 0])
        x_max_ap = np.max(ap_locations_array[:, 0])
        y_min_ap = np.min(ap_locations_array[:, 1])
        y_max_ap = np.max(ap_locations_array[:, 1])

        x_min = x_min_ap - buffer
        x_max = x_max_ap + buffer
        y_min = y_min_ap - buffer
        y_max = y_max_ap + buffer

    x_min_fixed = round_to_half(x_min)
    y_min_fixed = round_to_half(y_min)
    
    x_max_fixed = round_to_half(x_max)
    y_max_fixed = round_to_half(y_max)
    
    x_min, x_max = x_min_fixed, x_max_fixed
    y_min, y_max = y_min_fixed, y_max_fixed
    
    x_limits = [x_min, x_max]
    y_limits = [y_min, y_max]
    
    # 4. Generate Grid Points
    x_coords = np.arange(x_min, x_max + resolution, resolution)
    y_coords = np.arange(y_min, y_max + resolution, resolution)
    
    # Calculate grid dimensions W (Width) and H (Height)
    W = len(x_coords)
    H = len(y_coords)

    # Create the meshgrid and flatten to N_points x 2 array
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    grid_points_tensor = torch.from_numpy(grid_points).float().cuda()

    print("\n--- Grid Generation Summary ---")
    print(f"X Bounds: [{x_min:.2f}, {x_max:.2f}] m")
    print(f"Y Bounds: [{y_min:.2f}, {y_max:.2f}] m")
    print(f"Grid Resolution: {resolution} m")
    print(f"Grid Dimensions (W x H): {W} x {H}")
    print(f"Total Reference Points: {grid_points.shape[0]}")
    
    # The function signature (Return type) only includes grid_points_tensor, x_limits, y_limits
    return grid_points_tensor, x_limits, y_limits, W, H

def get_csi_blocks_with_cache(hmatrix_folder: str, config: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Load CSI data using a cache file. If cache doesn't exist, parse CSVs and save as .npy.
    The cache file is stored inside the hmatrix_folder as 'cache.npy'.
    """
    # Define cache path inside the CSV folder
    cache_path = os.path.join(hmatrix_folder, "cache.npy")
    
    # --- Stage 1: Get Raw Data Matrix ---
    if os.path.exists(cache_path):
        print(f"[Cache] Loading existing cache from {cache_path}")
        raw_csi_tensor = np.load(cache_path)
    else:
        print(f"[Cache] No cache found. Processing CSVs in {hmatrix_folder}...")
        
        # Original parameters
        Q = len(config.get('ACCESS_POINTS', {}))
        T = config['NUM_SAMPLE']
        P = config['NUM_PACKET']
        N = config['CSI_DIMENSIONS']['NUM_RX_ANTENNAS']
        M = config['CSI_DIMENSIONS']['NUM_SUBCARRIERS']
        
        filename_pattern = re.compile(r't(\d+)_hmatrix\.txSet\d+\.txPt\d+\.rxSet(\d+)\.inst(\d+)\.csv')
        data_storage = {}
        ap_ids_in_dataset = set()

        # Step 1.1: Parse filenames and load CSVs
        for filename in os.listdir(hmatrix_folder):
            # Skip the cache file itself if it exists
            if filename == "cache.npy": continue
            
            match = filename_pattern.match(filename)
            if match:
                time_stamp, rx_id, inst_id = int(match.group(1)), int(match.group(2)), int(match.group(3))
                ap_ids_in_dataset.add(rx_id)
                
                # _read_antenna_data_csv must exist in your utils.py
                h_complex = _read_antenna_data_csv(os.path.join(hmatrix_folder, filename), N)
                
                if h_complex is not None:
                    data_storage.setdefault(time_stamp, {}).setdefault(rx_id, {})[inst_id] = h_complex

        # Step 1.2: Align data and aggregate
        sorted_ap_ids = sorted(list(ap_ids_in_dataset))
        all_time_stamps = sorted(list(data_storage.keys()))
        total_samples = len(all_time_stamps)
        
        if total_samples == 0:
            return []

        raw_csi_tensor = np.zeros((total_samples, Q, N, M), dtype=np.complex64)
        for i, ts in enumerate(all_time_stamps):
            for j, rx_id in enumerate(sorted_ap_ids):
                ap_data = data_storage[ts].get(rx_id, {})
                for inst_id in range(1, M + 1):
                    if inst_id in ap_data:
                        raw_csi_tensor[i, j, :, inst_id - 1] = ap_data[inst_id]

        # Step 1.3: Save to cache
        np.save(cache_path, raw_csi_tensor)
        print(f"[Cache] New cache created at {cache_path}")

    # --- Stage 2: Dimension Transformation (Trim -> Reshape -> Transpose) ---
    T, P = config['NUM_SAMPLE'], config['NUM_PACKET']
    TP_block_size = T * P
    num_blocks = raw_csi_tensor.shape[0] // TP_block_size

    if num_blocks == 0:
        print(f"[Error] Total samples ({raw_csi_tensor.shape[0]}) < block size ({TP_block_size}).")
        return []

    # Final shape target: (Num_Blocks, Q, TP, N, M)
    trimmed = raw_csi_tensor[:num_blocks * TP_block_size]
    # Reshape to (Num_Blocks, TP, Q, N, M)
    reshaped = trimmed.reshape(num_blocks, TP_block_size, raw_csi_tensor.shape[1], raw_csi_tensor.shape[2], raw_csi_tensor.shape[3])
    # Transpose to (Num_Blocks, Q, TP, N, M)
    final_tensor = reshaped.transpose(0, 2, 1, 3, 4)

    # Convert to List of Tensors
    return [torch.from_numpy(block).to(torch.complex64) for block in final_tensor]

def _read_antenna_data_csv(file_path: str, N: int) -> Optional[np.ndarray]:
    """
    Read a single CSI CSV file containing data for one subcarrier
    across N receive antennas, and convert it into a complex-valued
    NumPy array of shape (N,).

    Args:
        file_path: Path to the CSI CSV file.
        N: Expected number of receive antennas (e.g., N=3).

    Returns:
        np.ndarray: Complex-valued array of shape (N,). Returns None if reading fails.
    """
    try:
        # Key modification: skip the first 4 lines of comments/header
        df = pd.read_csv(
            file_path, 
            skiprows=4,      
            skipinitialspace=True, 
            header=None,     
            index_col=False
        )
        
        # Ensure there are at least 4 columns (Rx Point, Rx Element, H_Real, H_Imag)
        if df.shape[1] < 4:
            print(f"Warning: {file_path} 數據列數不足。")
            return None
        df = df.iloc[:, :4] 
        
        # Manually assign column name
        df.columns = ['Rx Point', 'Rx Element', 'H_Real', 'H_Imag']
        
        # Data validation: number of rows should match the number of antennas
        if len(df) != N:
            print(f"Warning: {file_path} 數據行數為 {len(df)}，與預期的 {N} 根天線不符。跳過檔案。")
            return None
        
        # Sort by antenna index to ensure consistent antenna ordering (1, 2, 3, ...)
        df = df.sort_values(by='Rx Element')

        # Construct complex CSI: H = H_real + j * H_imag
        H_complex = (df['H_Real'].values + 1j * df['H_Imag'].values).astype(np.complex64)
        
        return H_complex 

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_history_coords_batch(
    reference_grid: torch.Tensor, 
    ref_index: int, 
    path: torch.Tensor, 
) -> torch.Tensor:
    
    G, T, _ = path.shape
    device = path.device

    FILL_VALUE = -torch.inf
    
    # Extract grid index sequence
    grid_indices_sequence = path[:, :, ref_index].clone()
    valid_mask = (grid_indices_sequence != -1)
    valid_indices_flat = grid_indices_sequence[valid_mask].long()
    
    # Coordinate lookup
    coords_flat = reference_grid[valid_indices_flat].to(device)
    
    # Initialize the output tensor with the special fill value for padding
    history_coords = torch.full(
        (G, T, 2), 
        FILL_VALUE, 
        dtype=torch.float32, 
        device=device
    )
    
    # Fill in the valid coordinates using the mask
    history_coords[valid_mask] = coords_flat
    
    return history_coords

def set_global_seed(seed: int) -> None:
    import os
    import random
    import numpy as np
    import torch

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # For older PyTorch versions that do not support warn_only.
        # Do not force strict mode here, to avoid breaking unsupported operations.
        pass
    except Exception as e:
        print(f"[Seed] Deterministic algorithm setting warning: {e}")

    print(f"[Seed] Fixed random seed enabled: {seed}")


def apply_reproducibility_config(config: dict) -> None:
    fix_random_seed = bool(config.get("FIX_RANDOM_SEED", False))

    if not fix_random_seed:
        print("[Seed] FIX_RANDOM_SEED=False. Random seed is not fixed.")
        return

    if "RANDOM_SEED" not in config:
        raise ValueError("FIX_RANDOM_SEED=True but RANDOM_SEED is not set in config.")

    set_global_seed(int(config["RANDOM_SEED"]))