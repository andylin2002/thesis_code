import torch
from .grid_tools import convert_grid_indices_to_coords

class ViterbiPathManager:
    def __init__(self, G: int, T: int, reference_grid: torch.Tensor, device: torch.device):

        self.G = G
        self.T = T
        self.reference_grid = reference_grid
        self.device = device
        
        # Initialize backpointer matrix to store parent indices
        self.bp = torch.full((G, T), -1, dtype=torch.long, device=device)

    def update(self, t: int, best_prev_indices: torch.Tensor):

        # Record the best previous node index for time t
        self.bp[:, t] = best_prev_indices

    def get_history_coords(self, end_t: int) -> torch.Tensor:

        # Create a container for indices with the exact valid length (0 to end_t)
        path_indices = torch.zeros((self.G, end_t + 1), dtype=torch.long, device=self.device)

        # Set the current grid points as the end of the trajectories
        current_indices = torch.arange(self.G, device=self.device)
        path_indices[:, end_t] = current_indices

        # Vectorized backtracking: trace from end_t down to 1
        temp_indices = current_indices
        for t in range(end_t, 0, -1):
            parents = self.bp[temp_indices, t]  # Lookup parent indices
            path_indices[:, t-1] = parents      # Store parent at previous step
            temp_indices = parents              # Move pointer backward

        # Map indices to physical coordinates (G, seq_len, 2)
        valid_coords = convert_grid_indices_to_coords(path_indices, self.reference_grid)
        
        return valid_coords

    def traceback_final_trajectory(self, end_t: int, end_node_index: torch.Tensor) -> torch.Tensor:

        # Container for the final best trajectory indices
        traj_indices = torch.zeros(self.T, dtype=torch.long, device=self.device)

        # Start from the best end node
        curr = end_node_index
        traj_indices[end_t] = curr
        
        # Backtrack one by one to t=0
        for t in range(end_t, 0, -1):
            parent = self.bp[curr, t]
            traj_indices[t-1] = parent
            curr = parent
            
        return self.reference_grid[traj_indices]