import torch
import numpy as np
import scipy.spatial.distance
from typing import Dict, Any, Tuple
from .model import QuantizedTF

def create_transformer_instance(
    config: Dict[str, Any], 
    N_DIRECTIONS: int, 
    device: torch.device
):
    
    D_MODEL = config['EMB_SIZE']
    N_HEADS = config['HEADS']
    N_LAYERS = config['LAYERS']
    DROPOUT = config['DROPOUT']
    D_FF = config['D_FF']

    model = QuantizedTF(
                enc_inp_size=2, 
                dec_inp_size=(N_DIRECTIONS + 1),
                dec_out_size=N_DIRECTIONS,
                
                layer=N_LAYERS,
                d_model=D_MODEL,
                d_ff=D_FF,
                h=N_HEADS,
                dropout=DROPOUT
            ).to(device)
    
    return model

def convert_long_trajectory_to_ids(
    batch_traj: torch.Tensor, directions_vectors: np.ndarray, config: Dict[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    T = batch_traj.shape[1]
    B = batch_traj.shape[0]

    inp_coords = batch_traj[:, :(T - 1), :].clone()
    
    velocities_abs = batch_traj[:, 1:, :] - batch_traj[:, :-1, :]
    velocities_flat = velocities_abs.cpu().reshape(-1, 2).numpy()
    target_ids_flat = scipy.spatial.distance.cdist(velocities_flat, directions_vectors).argmin(axis=1)
    target_ids = torch.tensor(target_ids_flat, dtype=torch.long, device=device).reshape(B, (T - 1))

    return inp_coords, target_ids