import torch
import torch.nn as nn
from multiprocessing import Queue
from typing import Dict, Any
import time
from .model import QuantizedTF 

# (TODO)
def create_transformer_instance(config: Dict[str, Any], device: torch.device):
    
    N_CLUSTERS = config.get('N_CLUSTERS', 50) 
    SOS_TOKEN = N_CLUSTERS
    
    model = QuantizedTF(
        N_CLUSTERS, 
        SOS_TOKEN + 1, 
        N_CLUSTERS, 
        N=config.get('LAYERS', 6),
        d_model=config.get('EMB_SIZE', 512),
        h=config.get('HEADS', 8)
    ).to(device)
    
    return model