# utils.py

import yaml
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import re

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
    ) -> Tuple[torch.Tensor, List[float], List[float]]:
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
    x_min_ap = np.min(ap_locations_array[:, 0])
    x_max_ap = np.max(ap_locations_array[:, 0])
    y_min_ap = np.min(ap_locations_array[:, 1])
    y_max_ap = np.max(ap_locations_array[:, 1])

    # Apply buffer distance (Final Localization Area)
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

def load_and_preprocess_csi_dataset(
    Hmatrix: str, 
    config: Dict[str, Any]
) -> List[torch.Tensor]:
    
    Q = len(config.get('ACCESS_POINTS', {}))
    T = config['NUM_SAMPLE']
    P = config['NUM_PACKET']
    N = config['CSI_DIMENSIONS']['NUM_RX_ANTENNAS']
    M = config['CSI_DIMENSIONS']['NUM_SUBCARRIERS']

    filename_pattern = re.compile(r't(\d+)_hmatrix\.txSet\d+\.txPt\d+\.rxSet(\d+)\.inst(\d+)\.csv')

    data_storage = {}
    ap_ids_in_dataset = set()
    
    # Step 1: Parse filenames and load CSI data
    for filename in os.listdir(Hmatrix):
        match = filename_pattern.match(filename)
        if match:
            time_stamp = int(match.group(1))
            rx_id = int(match.group(2))
            inst_id = int(match.group(3)) # Subcarrier ID
            ap_ids_in_dataset.add(rx_id)
            
            file_path = os.path.join(Hmatrix, filename)
            
            # Read CSI data of a single subcarrier across N receive antennas (shape: N_rx,)
            H_complex_N = _read_antenna_data_csv(file_path, N)
            
            if H_complex_N is not None:
                if time_stamp not in data_storage:
                    data_storage[time_stamp] = {}
                if rx_id not in data_storage[time_stamp]:
                    data_storage[time_stamp][rx_id] = {}
                    
                data_storage[time_stamp][rx_id][inst_id] = H_complex_N

    # Step 2: Align data and aggregate along the subcarrier (M) dimension
    sorted_ap_ids = sorted(list(ap_ids_in_dataset))
    all_time_stamps = sorted(list(data_storage.keys()))
    total_samples = len(all_time_stamps)
    
    if total_samples == 0:
        return []

    # Final raw CSI tensor shape: (Total_Samples, Q_ap, N_rx, M_sub)
    raw_csi_tensor = np.zeros((total_samples, Q, N, M), dtype=np.complex64)
    
    # Assume subcarrier indices range from 1 to M_sub
    subcarrier_indices = list(range(1, M + 1)) 

    for i, ts in enumerate(all_time_stamps):
        for j, rx_id in enumerate(sorted_ap_ids):
            ap_data = data_storage[ts].get(rx_id, {})
            
            # Stack data for each subcarrier ID (1 to M)
            for k, inst_id in enumerate(subcarrier_indices):
                if inst_id in ap_data:
                    # H_complex_N has shape (N_rx,)
                    raw_csi_tensor[i, j, :, k] = ap_data[inst_id] 
                # Otherwise, keep zero (missing subcarrier data)
    
    # Step 3: Block segmentation and reshaping
    TP_block_size = T * P
    num_blocks = total_samples // TP_block_size

    if num_blocks == 0:
        print(f"Error: Total samples ({total_samples}) is less than required block size (T*P={TP_block_size}).")
        return []

    # Reshape and permute axes:
    # (Num_Blocks, Q, TP, N, M)
    trimmed_tensor = raw_csi_tensor[:num_blocks * TP_block_size]
    reshaped_tensor = trimmed_tensor.reshape(num_blocks, TP_block_size, Q, N, M)
    
    # (Num_Blocks, TP, Q, N, M) -> (Num_Blocks, Q, TP, N, M)
    final_tensor = reshaped_tensor.transpose(0, 2, 1, 3, 4)

    # Convert to a list of PyTorch tensors (kept on CPU)
    csi_blocks_list = [
        torch.from_numpy(block).to(torch.complex64)
        for block in final_tensor
    ]
    
    print(f"Successfully generated {len(csi_blocks_list)} CSI blocks, each of shape ({Q}, {TP_block_size}, {N}, {M}).")
    
    return csi_blocks_list

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


