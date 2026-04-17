# engines/neural_engine/stages/proxy.py

from __future__ import annotations

from typing import Any, Dict, List

import torch

from engines.symbolic_engine.stages.result_estimation.proposed import soft_em_utils


class ProxyStage:
    """Build long-horizon expected-logprob targets for window batches."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.eps = 1e-8

    def build(
        self,
        emission_log_probs_qgt: torch.Tensor,
        neighbor_index_matrix: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Input:
            emission_log_probs_qgt: [S, W, Q, G, T]
            neighbor_index_matrix: [G, K]

        Output:
            {
                "target_score": [S, W, Q, T],
                "posterior_gt_long": [S, G, W*T],
                "expected_logprob_long": [S, Q, W*T],
            }
        """
        if emission_log_probs_qgt.ndim != 5:
            raise ValueError(
                f"Expected emission_log_probs_qgt shape [S,W,Q,G,T], got {tuple(emission_log_probs_qgt.shape)}"
            )

        if neighbor_index_matrix.ndim != 2:
            raise ValueError(
                f"Expected neighbor_index_matrix shape [G,K], got {tuple(neighbor_index_matrix.shape)}"
            )

        if emission_log_probs_qgt.device != self.device:
            emission_log_probs_qgt = emission_log_probs_qgt.to(self.device, non_blocking=True)

        if neighbor_index_matrix.device != self.device:
            neighbor_index_matrix = neighbor_index_matrix.to(self.device, non_blocking=True)

        num_windows, window_size, num_aps, num_grids, num_steps = emission_log_probs_qgt.shape
        neighbor_grids, _ = neighbor_index_matrix.shape

        if num_grids != neighbor_grids:
            raise ValueError(
                f"Grid size mismatch: emission G={num_grids}, neighbor G={neighbor_grids}"
            )

        target_score_list: List[torch.Tensor] = []
        posterior_gt_long_list: List[torch.Tensor] = []
        expected_logprob_long_list: List[torch.Tensor] = []

        for s in range(num_windows):
            emission_wqgt = emission_log_probs_qgt[s]                           # [W,Q,G,T]
            long_emission_qgt = self._build_long_emission(emission_wqgt)        # [Q,G,W*T]
            long_agg_gt = long_emission_qgt.sum(dim=0)                          # [G,W*T]

            posterior_gt_long = soft_em_utils.run_forward_backward(
                emission_log_probs=long_agg_gt,
                neighbor_index_matrix=neighbor_index_matrix,
                device=self.device,
            )  # [G,W*T]

            posterior_gt_long = self._normalize_posterior(posterior_gt_long)

            expected_logprob_long = self._compute_expected_logprob(
                long_emission_qgt,
                posterior_gt_long,
            )  # [Q,W*T]

            target_score = self._split_long_score_to_blocks(
                score_qt=expected_logprob_long,
                window_size=window_size,
                num_steps=num_steps,
            )  # [W,Q,T]

            target_score_list.append(target_score)
            posterior_gt_long_list.append(posterior_gt_long)
            expected_logprob_long_list.append(expected_logprob_long)

        return {
            "target_score": torch.stack(target_score_list, dim=0),                  # [S,W,Q,T]
            "posterior_gt_long": torch.stack(posterior_gt_long_list, dim=0),        # [S,G,W*T]
            "expected_logprob_long": torch.stack(expected_logprob_long_list, dim=0),# [S,Q,W*T]
        }

    def _build_long_emission(
        self,
        emission_wqgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        [W, Q, G, T] -> [Q, G, W*T]
        """
        w, q, g, t = emission_wqgt.shape
        return (
            emission_wqgt.permute(1, 2, 0, 3)
            .contiguous()
            .reshape(q, g, w * t)
        )

    def _normalize_posterior(
        self,
        posterior_gt: torch.Tensor,
    ) -> torch.Tensor:
        posterior_gt = posterior_gt.clamp_min(0.0)
        posterior_sum = posterior_gt.sum(dim=0, keepdim=True).clamp_min(self.eps)
        return posterior_gt / posterior_sum

    def _compute_expected_logprob(
        self,
        emission_qgt: torch.Tensor,
        posterior_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        emission_qgt: [Q, G, T_total]
        posterior_gt: [G, T_total]
        return: [Q, T_total]
        """
        return (emission_qgt * posterior_gt.unsqueeze(0)).sum(dim=1)

    def _split_long_score_to_blocks(
        self,
        score_qt: torch.Tensor,
        window_size: int,
        num_steps: int,
    ) -> torch.Tensor:
        """
        [Q, W*T] -> [W, Q, T]
        """
        num_aps, total_steps = score_qt.shape
        expected_total = window_size * num_steps

        if total_steps != expected_total:
            raise ValueError(
                f"Expected total_steps={expected_total}, got {total_steps}"
            )

        return (
            score_qt.reshape(num_aps, window_size, num_steps)
            .permute(1, 0, 2)
            .contiguous()
        )