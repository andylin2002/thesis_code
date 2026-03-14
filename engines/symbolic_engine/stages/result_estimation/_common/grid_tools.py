# engines/symbolic_engine/stages/result_estimation/_common/grid_tools.py

import torch
import numpy as np
from typing import Dict, Any

def get_ap_locations(config: Dict[str, Any], Q: int, device: torch.Tensor) -> torch.Tensor:
    """
    Extract AP locations from config and convert to Tensor.
    """
    ap_locations_list = []
    for ap_id in range(1, Q + 1):
        ap_key = f"AP_{ap_id}"
        if ap_key in config['ACCESS_POINTS']:
            ap_locations_list.append(config['ACCESS_POINTS'][ap_key]['LOCATION_M'])
        else:
            raise ValueError(f"Missing location data for AP ID {ap_id}")
        
    ap_locations = torch.tensor(ap_locations_list, dtype=torch.float32, device=device)

    return ap_locations

def calculate_L_gq(
        reference_grid: torch.Tensor, 
        ap_locations: torch.Tensor, 
) -> torch.Tensor:
    """
    Calculate Log10 Distance from every Grid point to every AP.
    Returns: (G, Q)
    """
    ap_locations_expanded = ap_locations.unsqueeze(0)     # [1, Q, 2]
    grid_expanded = reference_grid.unsqueeze(1)           # [G, 1, 2]

    squared_diff = (ap_locations_expanded - grid_expanded) ** 2
    distance_matrix = torch.sqrt(squared_diff.sum(dim=2))
    distance_matrix = torch.clamp(distance_matrix, min=1e-10)
    L_gq = torch.log10(distance_matrix)

    return L_gq

def calculate_angle_gq_mean(
        reference_grid: torch.Tensor, 
        ap_locations: torch.Tensor, 
        ap_orientations: torch.Tensor
) -> torch.Tensor:
    """
    Calculate Geometric Angle from every Grid point to every AP.
    Returns: (G, Q)
    """
    ap_locations_expanded = ap_locations.unsqueeze(0)     # [1, Q, 2]
    grid_expanded = reference_grid.unsqueeze(1)           # [G, 1, 2]

    vector_V_gq = grid_expanded - ap_locations_expanded
    x_diff = vector_V_gq[..., 0]
    y_diff = vector_V_gq[..., 1]

    angle_rad_gq = torch.atan2(y_diff, x_diff)
    angle_deg_gq = torch.rad2deg(angle_rad_gq)

    angle_deg_gq = torch.fmod(angle_deg_gq + 360.0, 360.0)

    ap_orientations_expanded = ap_orientations.unsqueeze(0).expand_as(angle_deg_gq) # [Q, T]
    diff = ap_orientations_expanded - angle_deg_gq
    relative_angle_deg = torch.remainder(diff + 180.0, 360.0) - 180.0
    angle_gq_mean = torch.clamp(relative_angle_deg, min=-90.0, max=90.0)

    return angle_gq_mean

def get_all_neighbor_indices(
    config: Dict[str, Any],
    grid_indices_G: torch.Tensor, 
    device: torch.device
) -> torch.Tensor:
    """
    Find 8-neighbors for Viterbi transition.
    """
    # Parameter Setup
    W = config['X_WIDTH'] 
    H = config['Y_WIDTH'] 
    G = grid_indices_G.shape[0] # Numbers of Reference Point
    
    # grid_indices_G must be of long type
    grid_indices_G = grid_indices_G.long() 
    
    # [G] -> [G, 1]
    current_rows_G = (grid_indices_G // W).unsqueeze(1) 
    current_cols_G = (grid_indices_G % W).unsqueeze(1)
    
    # Offset matrix (3x3 = 9 combinations), shape [1, 9]
    offset_row_1_9 = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.long, device=device).unsqueeze(0)
    offset_col_1_9 = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.long, device=device).unsqueeze(0)
  
    # Broadcasting [G, 1] + [1, 9] -> [G, 9]
    candidate_rows_G_9 = current_rows_G + offset_row_1_9
    candidate_cols_G_9 = current_cols_G + offset_col_1_9
    
    # Boundary Check (using mask), shape [G, 9]
    mask_row_valid = (candidate_rows_G_9 >= 0) & (candidate_rows_G_9 < H)
    mask_col_valid = (candidate_cols_G_9 >= 0) & (candidate_cols_G_9 < W)
    valid_mask_G_9 = mask_row_valid & mask_col_valid
    
    # Convert 2D candidate coordinates to 1D indices, applying the validity mask
    all_candidate_indices_G_9 = candidate_rows_G_9 * W + candidate_cols_G_9
    neighbor_indices = torch.full((G, 9), -1, dtype=torch.long, device=device)
    neighbor_indices[valid_mask_G_9] = all_candidate_indices_G_9[valid_mask_G_9]
    
    return neighbor_indices

def convert_grid_indices_to_coords(indices: torch.Tensor, reference_grid: torch.Tensor) -> torch.Tensor:
    """
    Convert grid indices back to (x, y) coordinates.
    Useful for generating final trajectory output.
    """
    return reference_grid[indices.long()]