# engines/symbolic_engine/stages/gating_evaluation/evaluator.py

import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from engines.neural_engine.stages.represent import RepresentStage
from engines.neural_engine.stages.train import CrossAPReliabilityHead


class GatingEvaluator:
    """
    Symbolic-side neural gating evaluator.

    This evaluator reuses the neural-side RepresentStage and
    CrossAPReliabilityHead directly, so future changes only need to be made in:
        - engines/neural_engine/stages/represent.py
        - engines/neural_engine/stages/train.py

    Input:
        raw_csi_block: [Q, T, N, M]

    Output:
        reliability_qt: [Q, T]
        logits_qt: [Q, T] (optional)
    """

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

        self.num_aps = len(config["ACCESS_POINTS"])
        self.num_rx_antennas = int(config["CSI_DIMENSIONS"]["NUM_RX_ANTENNAS"])
        self.num_subcarriers = int(config["CSI_DIMENSIONS"]["NUM_SUBCARRIERS"])
        self.latent_dim = int(config.get("LATENT_DIM", 128))

        # Reuse neural-side feature builder + encoder through RepresentStage
        self.represent = RepresentStage(config, device)

        # Reuse the exact same reliability head definition as training side
        self.reliability_head = CrossAPReliabilityHead(
            feature_dim=self.latent_dim,
            hidden_dim=int(config.get("RELIABILITY_HEAD_HIDDEN", 64)),
            dropout=float(config.get("RELIABILITY_HEAD_DROPOUT", 0.1)),
        ).to(self.device)

        self.encoder = self.represent.encoder
        self._set_eval_mode()

    def _set_eval_mode(self) -> None:
        self.encoder.eval()
        self.reliability_head.eval()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict)}")

        encoder_state = state_dict.get("encoder")
        reliability_head_state = state_dict.get("reliability_head")

        if encoder_state is None:
            raise KeyError("Missing key 'encoder' in gating state_dict")
        if reliability_head_state is None:
            raise KeyError("Missing key 'reliability_head' in gating state_dict")

        self.encoder.load_state_dict(encoder_state, strict=True)
        self.reliability_head.load_state_dict(reliability_head_state, strict=True)
        self._set_eval_mode()

        print("[GatingEvaluator] Neural weights hot-swapped successfully.")

    @torch.no_grad()
    def evaluate(
        self,
        raw_csi_block: torch.Tensor,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            raw_csi_block: [Q, T, N, M]

        Returns:
            reliability_qt: [Q, T]
            logits_qt: [Q, T] if return_logits=True
        """
        if not isinstance(raw_csi_block, torch.Tensor):
            raise TypeError(
                f"AI Gating fail: raw_csi_block must be torch.Tensor, got {type(raw_csi_block)}"
            )

        if raw_csi_block.device != self.device:
            raw_csi_block = raw_csi_block.to(self.device, non_blocking=True)

        if raw_csi_block.ndim != 4:
            raise ValueError(
                f"AI Gating fail: expected raw_csi_block shape [Q, T, N, M], got {tuple(raw_csi_block.shape)}"
            )

        num_aps, num_steps, num_antennas, num_subcarriers = raw_csi_block.shape

        if num_aps != self.num_aps:
            raise ValueError(
                f"AI Gating fail: configured num_aps={self.num_aps}, but got {num_aps}"
            )
        if num_antennas != self.num_rx_antennas:
            raise ValueError(
                f"AI Gating fail: configured num_rx_antennas={self.num_rx_antennas}, but got {num_antennas}"
            )
        if num_subcarriers != self.num_subcarriers:
            raise ValueError(
                f"AI Gating fail: configured num_subcarriers={self.num_subcarriers}, but got {num_subcarriers}"
            )

        # [Q, T, N, M] -> [1, Q, T, N, M]
        raw_csi_batch = raw_csi_block.unsqueeze(0)

        # Reuse RepresentStage's exact feature-building logic
        input_features = self.represent._build_input_features(raw_csi_batch)  # [1, Q, T, C, M]

        # Reuse RepresentStage's exact encoder path
        encoded = self.represent._encode_features(input_features)             # [1, T, Q, D]

        if return_logits:
            reliability_btq, logits_btq = self.reliability_head(
                encoded,
                return_logits=True,
            )
        else:
            reliability_btq = self.reliability_head(
                encoded,
                return_logits=False,
            )
            logits_btq = None

        if reliability_btq.ndim != 3:
            raise RuntimeError(
                f"AI Gating fail: reliability head must output [B, T, Q], got {tuple(reliability_btq.shape)}"
            )

        reliability_qt = reliability_btq[0].transpose(0, 1).contiguous()  # [Q, T]

        if return_logits:
            if logits_btq is None:
                raise RuntimeError(
                    "AI Gating fail: return_logits=True but logits were not produced"
                )
            logits_qt = logits_btq[0].transpose(0, 1).contiguous()         # [Q, T]
            return reliability_qt, logits_qt

        return reliability_qt