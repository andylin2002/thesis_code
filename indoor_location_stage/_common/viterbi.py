import torch
from typing import Callable, Tuple, Optional

from .grid_tools import convert_grid_indices_to_coords

class Viterbi_Algorithm:
    def __init__(self, G: int, T: int, reference_grid: torch.Tensor, device: torch.device):
        """
        Initialize Viterbi container.
        """
        self.G = G
        self.T = T
        self.reference_grid = reference_grid
        self.device = device
        
        # Initialize backpointer matrix to store parent indices
        self.bp = torch.full((G, T), -1, dtype=torch.long, device=device)

    def run(
        self, 
        emission_log_probs: torch.Tensor, 
        neighbor_index_matrix: torch.Tensor,
        get_max_previous_score: Callable,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute Viterbi Forward-Backward algorithm.

        Args:
            emission_log_probs: Pre-calculated log probs (G, T).
            neighbor_index_matrix: Adjacency info (G, 9).
            transition_handler: Callback to calc transition probs. 
                                Sig: (neighbor_matrix, delta, history_coords, **kwargs) -> (indices, values)
            **transition_kwargs: Extra args for handler (e.g., model, mode, sos_token).

        Returns:
            trajectory: Optimal path coordinates (T, 2).
            max_log_prob: Total log probability of the path.
        """
        # 1. Initialization (t=0)
        delta = emission_log_probs[:, 0].clone()

        # 2. Forward Pass
        for t in range(1, self.T):
            current_emission_log_probs = emission_log_probs[:, t]

            # Calculate best transitions using external handler
            winner_indices, max_prev_path_scores = get_max_previous_score(
                neighbor_index_matrix, 
                delta,
                t, 
                **kwargs
            )

            # Update delta: Emission + Max(Prev_Delta + Transition)
            delta = current_emission_log_probs + max_prev_path_scores

            # Update backpointers
            self.bp[:, t] = winner_indices

        # 3. Find best end node
        max_log_prob, best_end_node = torch.max(delta, dim=0)

        # 4. Backward Pass
        trajectory = self._traceback_final_trajectory(self.T-1, best_end_node)
        
        return trajectory, max_log_prob

    def _traceback_final_trajectory(self, end_t: int, end_node_index: torch.Tensor) -> torch.Tensor:
        """
        Traceback the single best path from the optimal end node.
        """
        traj_indices = torch.zeros(self.T, dtype=torch.long, device=self.device)

        # Start from best end node
        curr = end_node_index
        traj_indices[end_t] = curr
        
        # Backtrack to t=0
        for t in range(end_t, 0, -1):
            parent = self.bp[curr, t]
            traj_indices[t-1] = parent
            curr = parent
            
        return self.reference_grid[traj_indices]