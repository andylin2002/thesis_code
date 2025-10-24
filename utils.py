import yaml
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import torch

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

def generate_reference_grid(config: Dict[str, Any]) -> Tuple[torch.Tensor, List[float], List[float]]:
    """
    Calculates the boundary of the localization area and generates
    a uniform grid of prediction points based on AP locations.

    Returns:
        tuple: (grid_points, x_limits, y_limits, ap_locations_array) 
               grid_points: Array of grid coordinates (N_points, 2).
               x_limits: [x_min, x_max] boundary.
               y_limits: [y_min, y_max] boundary.
               ap_locations_array: Array of AP coordinates (N_AP, 2).
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
        return np.array([]), [0.0, 0.0], [0.0, 0.0], np.array([]) 

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
    
    # Create the meshgrid and flatten to N_points x 2 array
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    grid_points_tensor = torch.from_numpy(grid_points).float().cuda()

    print("\n--- Grid Generation Summary ---")
    print(f"X Bounds: [{x_min:.2f}, {x_max:.2f}] m")
    print(f"Y Bounds: [{y_min:.2f}, {y_max:.2f}] m")
    print(f"Grid Resolution: {resolution} m")
    print(f"Total Reference Points: {grid_points.shape[0]}")
    
    return grid_points_tensor, x_limits, y_limits

def visualize_grid_and_aps(grid_points: np.ndarray, ap_locations: np.ndarray, x_bounds: List[float], y_bounds: List[float]):
    """
    Plots AP locations and generated grid points for visual verification.
    """
    if grid_points.size == 0:
        print("Cannot visualize: Grid is empty.")
        return

    plt.figure(figsize=(10, 8))
    
    # Plot Grid Points
    plt.scatter(grid_points[:, 0], grid_points[:, 1], 
                s=5, color='gray', alpha=0.5, label='Prediction Grid Points')
    
    # Plot APs
    plt.scatter(ap_locations[:, 0], ap_locations[:, 1], 
                s=100, color='red', marker='X', label='Access Points (APs)', zorder=5)
    
    # Label APs
    for i, (x, y) in enumerate(ap_locations):
        plt.text(x + 0.5, y, f'AP_{i+1}', fontsize=10, color='red')
        
    # Draw Bounding Box
    rect = plt.Rectangle((x_bounds[0], y_bounds[0]), 
                         x_bounds[1] - x_bounds[0], 
                         y_bounds[1] - y_bounds[0],
                         fill=False, edgecolor='blue', linestyle='--', linewidth=2, label='Localization Area')
    plt.gca().add_patch(rect)
    
    plt.title('Verification of Localization Grid Area')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.axis('equal') 
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def load_raw_csi(path):
    """Loads the integrated complex CSI data from a .npy file."""
    try:
        complex_csi = np.load(path)
        
        if complex_csi.shape != (4, 1500, 3, 30) or complex_csi.dtype != np.complex64:
            print(f"Error: Data shape or dtype mismatch in {path}.")
            print(f"Expected (1500, 4, 3, 30) and complex64, got {complex_csi.shape} and {complex_csi.dtype}.")
            return None
            
        return complex_csi
    
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {path}. Please check file existence.")
        return None