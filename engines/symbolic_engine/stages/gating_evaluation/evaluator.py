# engines/symbolic_engine/stages/gating_evaluation/evaluator.py

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from core.models.csi_encoder import CSIEncoder
from engines.neural_engine.stages.represent import CrossAPAttention


class GatingEvaluator(nn.Module):
    """
    GatingEvaluator
    ----------------
    Inference-only module that converts a raw CSI block into AP-wise reliability.

    Design:
        1. Use the same preprocessing as the neural RepresentStage
           (amplitude normalization + inter-antenna phase difference).
        2. Use the trained encoder + attention to infer AP probabilities p(q,t),
           where sum over APs = 1 for each time step t.
        3. Convert probabilities to reliability:
               reliability(q,t) = Q * p(q,t)
           so that:
               - if all APs are equally probable => reliability(q,t) = 1
               - more reliable APs => reliability(q,t) > 1
               - less reliable APs => reliability(q,t) < 1
        4. Output shape is aligned with symbolic engine:
               single block  -> [Q, T]
               batch blocks  -> [B, Q, T]

    Expected raw CSI input shape:
        single block: [Q, T, N, M]
        batch block : [B, Q, T, N, M]

    Notes:
        - This evaluator does NOT train.
        - This evaluator does NOT generate pseudo targets.
        - This evaluator only performs inference and probability->reliability conversion.
        - This evaluator must stay architecturally aligned with the neural training side.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.Q = len(config["ACCESS_POINTS"])
        self.N = int(config["CSI_DIMENSIONS"]["NUM_RX_ANTENNAS"])
        self.M = int(config["CSI_DIMENSIONS"]["NUM_SUBCARRIERS"])
        self.D = int(config.get("LATENT_DIM", 128))

        # Keep the evaluator aligned with training-side attention temperature.
        self.attention_temperature = float(config.get("ATTENTION_TEMPERATURE", 0.5))

        # Optional clamp range for reliability.
        # This is not the core mechanism; it is only a numerical safeguard.
        self.reliability_min = float(config.get("RELIABILITY_MIN", 0.0))
        self.reliability_max = float(config.get("RELIABILITY_MAX", float(self.Q)))

        # Shared encoder (same architecture as neural training side)
        self.encoder = CSIEncoder(
            num_antennas=self.N,
            num_subcarriers=self.M,
            latent_dim=self.D
        ).to(self.device)

        # Same context-aware cross-AP scorer as training side
        self.attention = CrossAPAttention(
            feature_dim=self.D,
            temperature=self.attention_temperature
        ).to(self.device)

        self.eval()

    # =========================================================
    # Public API
    # =========================================================
    @torch.no_grad()
    def evaluate(self, raw_csi: torch.Tensor, return_debug: bool = False):
        """
        Convert raw CSI into AP-wise reliability.

        Args:
            raw_csi:
                [Q, T, N, M]       for a single CSI block
                [B, Q, T, N, M]    for a batch of CSI blocks
            return_debug:
                False -> return reliability only
                True  -> return (reliability, debug_dict)

        Returns:
            reliability:
                [Q, T]       for single block
                [B, Q, T]    for batch blocks

            debug_dict (optional):
                {
                    "ap_probs": [Q, T] or [B, Q, T],
                    "scores":   [Q, T] or [B, Q, T],
                }
        """
        self.eval()

        if raw_csi.device != self.device:
            raw_csi = raw_csi.to(self.device, non_blocking=True)

        is_single = (raw_csi.dim() == 4)

        if is_single:
            raw_csi = raw_csi.unsqueeze(0)   # -> [1, Q, T, N, M]

        if raw_csi.dim() != 5:
            raise ValueError(
                f"[GatingEvaluator] Expected raw_csi with shape [Q,T,N,M] or [B,Q,T,N,M], "
                f"but got shape {tuple(raw_csi.shape)}"
            )

        B, Q, T, N, M = raw_csi.shape

        if Q != self.Q:
            raise ValueError(f"[GatingEvaluator] Q mismatch: expected {self.Q}, got {Q}")
        if N != self.N:
            raise ValueError(f"[GatingEvaluator] N mismatch: expected {self.N}, got {N}")
        if M != self.M:
            raise ValueError(f"[GatingEvaluator] M mismatch: expected {self.M}, got {M}")
        if N < 2:
            raise ValueError(
                "[GatingEvaluator] NUM_RX_ANTENNAS must be at least 2 "
                "to compute inter-antenna phase difference."
            )

        features = self._preprocess(raw_csi)  # [B, Q, T, N-1, M, 2]

        features = features.reshape(B * Q, T, N - 1, M, 2).contiguous()
        encoded = self.encoder(features, return_projection=False)  # [B*Q, T, D]
        encoded = encoded.view(B, Q, T, -1).permute(0, 2, 1, 3).contiguous()  # [B, T, Q, D]

        # now get both probs and raw logits
        ap_probs, scores = self.attention(encoded, return_scores=True)  # [B, T, Q], [B, T, Q]

        reliability = ap_probs * float(self.Q)  # [B, T, Q]
        reliability = reliability.clamp(self.reliability_min, self.reliability_max)

        # reorder to [B, Q, T]
        reliability = reliability.permute(0, 2, 1).contiguous()
        ap_probs_qt = ap_probs.permute(0, 2, 1).contiguous()
        scores_qt = scores.permute(0, 2, 1).contiguous()

        if is_single:
            reliability = reliability[0]   # [Q, T]
            ap_probs_qt = ap_probs_qt[0]   # [Q, T]
            scores_qt = scores_qt[0]       # [Q, T]

        if not return_debug:
            return reliability

        debug_dict = {
            "ap_probs": ap_probs_qt,
            "scores": scores_qt,
        }
        return reliability, debug_dict

    def load_state_dict(self, state_dict: Optional[dict], strict: bool = False):
        """
        Load encoder/attention weights published from the neural engine.

        Expected state_dict format from TrainStage.get_state_dict():
        {
            "encoder": {...},
            "attention": {...},
            "predictor": {...}   # ignored here
        }
        """
        if state_dict is None:
            return

        encoder_sd = state_dict.get("encoder", None)
        attention_sd = state_dict.get("attention", None)

        if encoder_sd is not None:
            self.encoder.load_state_dict(encoder_sd, strict=strict)

        if attention_sd is not None:
            self.attention.load_state_dict(attention_sd, strict=strict)

        self.eval()

    # =========================================================
    # Internal helpers
    # =========================================================
    def _preprocess(self, raw_csi: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw CSI exactly the same way as neural RepresentStage.

        Input:
            raw_csi: [B, Q, T, N, M] (complex tensor)

        Output:
            features: [B, Q, T, N-1, M, 2]
                channel 0: normalized amplitude
                channel 1: inter-antenna phase difference
        """
        if not torch.is_complex(raw_csi):
            raise TypeError(
                f"[GatingEvaluator] raw_csi must be a complex tensor, got dtype={raw_csi.dtype}"
            )

        amplitude = torch.abs(raw_csi)   # [B,Q,T,N,M]
        amplitude = amplitude / (amplitude.mean(dim=(-1, -2), keepdim=True) + 1e-8)

        phase = torch.angle(raw_csi)     # [B,Q,T,N,M]

        # Inter-antenna phase difference with wrapping for stability
        phase_delta = phase[:, :, :, 1:, :] - phase[:, :, :, :-1, :]
        phase_diff = torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta))

        # Match phase-diff antenna dimension: [N-1]
        amplitude = amplitude[:, :, :, 1:, :]   # [B,Q,T,N-1,M]

        features = torch.stack([amplitude, phase_diff], dim=-1)  # [B,Q,T,N-1,M,2]
        return features