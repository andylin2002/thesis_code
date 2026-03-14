# engines/symbolic_engine/stages/result_estimation/_common/viterbi.py

import torch
from typing import Callable, Tuple, Optional

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
        transition_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 0. Basic checks / alignment
        if emission_log_probs.dim() != 2:
            raise ValueError(f"emission_log_probs must be (G,T), got {tuple(emission_log_probs.shape)}")
        if emission_log_probs.size(0) != self.G or emission_log_probs.size(1) != self.T:
            raise ValueError(
                f"emission_log_probs shape mismatch: expected (G,T)=({self.G},{self.T}), got {tuple(emission_log_probs.shape)}"
            )
        if neighbor_index_matrix.dim() != 2 or neighbor_index_matrix.size(0) != self.G:
            raise ValueError(
                f"neighbor_index_matrix must be (G,K), got {tuple(neighbor_index_matrix.shape)}"
            )

        K = neighbor_index_matrix.size(1)

        use_tpd = transition_log_probs is not None
        if use_tpd:
            if transition_log_probs.dim() != 3:
                raise ValueError(f"transition_log_probs must be 3D, got {transition_log_probs.dim()}D")
            if transition_log_probs.size(0) != self.T - 1:
                raise ValueError(
                    f"transition_log_probs first dim must be T-1={self.T-1}, got {transition_log_probs.size(0)}"
                )
            if transition_log_probs.size(1) != self.G or transition_log_probs.size(2) != K:
                raise ValueError(
                    f"transition_log_probs shape mismatch: expected (T-1,G,K)=({self.T-1},{self.G},{K}), "
                    f"got {tuple(transition_log_probs.shape)}"
                )
            transition_log_probs = transition_log_probs.to(self.device).type_as(emission_log_probs)

        # 1. Initialization (t=0)
        delta = emission_log_probs[:, 0].clone()

        # --- Precompute adjacency validity (graph-level invariant) ---
        # neighbor_index_matrix: (G, K), -1 means invalid neighbor
        valid_mask = (neighbor_index_matrix != -1)          # (G, K)
        no_valid = ~valid_mask.any(dim=1)                   # (G,)
        row_indices = torch.arange(self.G, device=self.device)

        neigh_safe = neighbor_index_matrix.clamp(min=0)       # (G, K)
        neg_inf = -float('inf')

        # 2. Forward Pass
        for t in range(1, self.T):
            current_emission_log_probs = emission_log_probs[:, t]

            if not use_tpd:
                # === Backward-compatible behavior ===
                winner_indices, max_prev_path_scores = get_max_previous_score(
                    neighbor_index_matrix,
                    delta
                )
            else:
                # === New behavior: delta_prev[g'] + transition_log_probs[t-1, g, k] ===
                # Gather delta_prev at neighbors: (G, K)
                neigh_prev = delta[neigh_safe]  # (G, K)
                neigh_prev = torch.where(
                    valid_mask,
                    neigh_prev,
                    torch.full_like(neigh_prev, neg_inf)
                )

                # Transition log-scores for time step (t-1 -> t): (G, K)
                tpd_gk = transition_log_probs[t - 1]

                # Combine and take max over k
                score_gk = neigh_prev + tpd_gk
                max_prev_path_scores, rel_k = torch.max(score_gk, dim=1)  # (G,), (G,)

                winner_indices = neighbor_index_matrix[row_indices, rel_k]  # (G,)

            # =========================================================
            # [GUARD] If a node has no valid neighbors, force self-transition
            # Score uses previous delta[g] (i.e., transition log-prob = 0)
            # =========================================================
            if no_valid.any():
                winner_indices = winner_indices.clone()
                max_prev_path_scores = max_prev_path_scores.clone()

                winner_indices[no_valid] = row_indices[no_valid]
                max_prev_path_scores[no_valid] = delta[no_valid]

            # (Optional extra safety)
            # If some hook accidentally returns -1 for other reasons, also self-fix:
            invalid_winner = (winner_indices < 0) | (winner_indices >= self.G)
            if invalid_winner.any():
                winner_indices = winner_indices.clone()
                max_prev_path_scores = max_prev_path_scores.clone()
                winner_indices[invalid_winner] = row_indices[invalid_winner]
                max_prev_path_scores[invalid_winner] = delta[invalid_winner]

            # Update delta: Emission + Max(Prev_Delta + Transition)
            delta = current_emission_log_probs + max_prev_path_scores

            # Update backpointers
            self.bp[:, t] = winner_indices

        # 3. Find best end node
        max_log_prob, best_end_node = torch.max(delta, dim=0)

        # 4. Backward Pass
        trajectory = self._traceback_final_trajectory(self.T - 1, best_end_node)

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