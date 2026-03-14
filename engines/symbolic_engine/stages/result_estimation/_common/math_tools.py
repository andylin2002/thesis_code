# engines/symbolic_engine/stages/result_estimation/_common/math_tools.py

import torch
import math
from typing import Dict, Any, Optional

from . import grid_tools

def gaussian_log_pdf(
        x: torch.Tensor, 
        mean: torch.Tensor, 
        variance: torch.Tensor, 
) -> torch.Tensor:
    """
    Calculate the Log Probability Density of a Gaussian distribution.
    """
    MIN_VAR_FOR_MODEL = 1e-4
    variance = torch.clamp(variance, min=MIN_VAR_FOR_MODEL)

    log_prob = -0.5 * torch.log(2.0 * torch.pi * variance) - 0.5 * (x - mean)**2 / variance

    return log_prob

def calculate_weighted_average(
        data: torch.Tensor, 
        weights: torch.Tensor, 
        dim: int = -1
    ) -> torch.Tensor:
    """
    Calculate the weighted average along a specific dimension.

    Args:
        data: The values to average.
        weights: The weights must have same shape as data or be broadcastable.
        dim: The dimension to reduce. Default is last dimension.
    """
    # Weighted Sum
    weighted_sum = (data * weights).sum(dim=dim)
    
    # Sum of Weights
    sum_weights = weights.sum(dim=dim)
    
    # Avoid division by zero
    sum_weights = torch.clamp(sum_weights, min=1e-10)

    return weighted_sum / sum_weights

def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the shortest difference between two angles in degrees.
    Handles the cyclic nature (e.g., diff between 359 and 1 is 2, not 358).
    
    Range: [-180, 180]
    """
    diff = pred - target
    # Map to [-180, 180]
    diff = torch.remainder(diff + 180.0, 360.0) - 180.0
    return diff

# TODO:目前還沒用到這個，proposed才會用到
def weighted_angular_average(
    angles: torch.Tensor, 
    weights: torch.Tensor, 
    dim: int = -1
) -> torch.Tensor:
    """
    Calculate the weighted average of angles.
    Standard averaging fails for angles (avg of 359 and 1 should be 0, not 180).
    We use vector components (sin/cos) to calculate this.
    """
    # Convert to radians
    angles_rad = torch.deg2rad(angles)
    
    # Decompose to components
    sin_val = torch.sin(angles_rad)
    cos_val = torch.cos(angles_rad)
    
    # Calculate weighted average of components
    avg_sin = calculate_weighted_average(sin_val, weights, dim)
    avg_cos = calculate_weighted_average(cos_val, weights, dim)
    
    # Convert back to degrees
    avg_angle_rad = torch.atan2(avg_sin, avg_cos)
    avg_angle_deg = torch.rad2deg(avg_angle_rad)
    
    # Map to [0, 360) or [-180, 180] as needed, here we prefer standard mod
    avg_angle_deg = torch.fmod(avg_angle_deg + 360.0, 360.0)
    
    return avg_angle_deg