# engines/neural_engine/stages/proxy.py

import torch
import torch.nn.functional as F
from typing import Any, Dict


class ProxyStage:
    """Build expected-logprob proxy from symbolic outputs."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.eps = 1e-8

    def build(
        self,
        emission_log_probs_qgt: torch.Tensor,
        posterior_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Input:
            emission_log_probs_qgt: [B, Q, G, T]
            posterior_gt: [B, G, T]

        Output:
            proxy_pkg = {
                "proxy": [B, Q, T],
                "expected_logprob": [B, Q, T],
            }
        """
        if emission_log_probs_qgt.ndim != 4:
            raise ValueError(
                f"Expected emission_log_probs_qgt with shape [B, Q, G, T], got {tuple(emission_log_probs_qgt.shape)}"
            )

        if posterior_gt.ndim != 3:
            raise ValueError(
                f"Expected posterior_gt with shape [B, G, T], got {tuple(posterior_gt.shape)}"
            )

        batch_size, num_aps, num_grids, num_steps = emission_log_probs_qgt.shape
        posterior_b, posterior_g, posterior_t = posterior_gt.shape

        if posterior_b != batch_size:
            raise ValueError(
                f"Batch size mismatch: emission={batch_size}, posterior={posterior_b}"
            )
        if posterior_g != num_grids:
            raise ValueError(
                f"Grid size mismatch: emission={num_grids}, posterior={posterior_g}"
            )
        if posterior_t != num_steps:
            raise ValueError(
                f"Time size mismatch: emission={num_steps}, posterior={posterior_t}"
            )

        # Normalize posterior if needed
        posterior = posterior_gt.clamp_min(0.0)
        posterior_sum = posterior.sum(dim=1, keepdim=True).clamp_min(self.eps)
        posterior = posterior / posterior_sum

        # Expected log-probability under posterior
        expected_logprob = (
            emission_log_probs_qgt * posterior.unsqueeze(1)
        ).sum(dim=2)  # [B, Q, T]

        # Convert score to AP-wise reliability weights
        proxy = num_aps * F.softmax(expected_logprob, dim=1)  # [B, Q, T]

        return {
            "proxy": proxy,
            "expected_logprob": expected_logprob,
        }