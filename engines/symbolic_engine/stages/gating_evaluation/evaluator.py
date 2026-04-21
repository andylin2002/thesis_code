# engines/symbolic_engine/stages/gating_evaluation/evaluator.py

import os
import torch
from typing import Any, Dict, Tuple, Union

from engines.neural_engine.stages.represent import RepresentStage
from core.models.model import NeuralReliabilityModel


class GatingEvaluator:
    """Symbolic-side neural gating evaluator."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

        self.num_aps = len(config["ACCESS_POINTS"])
        self.num_rx_antennas = int(config["CSI_DIMENSIONS"]["NUM_RX_ANTENNAS"])
        self.num_subcarriers = int(config["CSI_DIMENSIONS"]["NUM_SUBCARRIERS"])

        self.represent = RepresentStage(config, device)
        self.model = NeuralReliabilityModel(config).to(self.device)
        self._set_eval_mode()
        self._load_checkpoint_if_exists()

    def _set_eval_mode(self) -> None:
        self.model.eval()

    def _load_checkpoint_if_exists(self) -> None:
        ckpt_dir = "checkpoint"
        scene_name = self.config.get("SCENARIO_NAME", "default_scene")
        ckpt_path = os.path.join(ckpt_dir, f"{scene_name}.ckpt")

        if not os.path.exists(ckpt_path):
            print(f"[GatingEvaluator] No checkpoint found at startup: {ckpt_path}")
            return

        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint["state_dict"]
            model_state = state_dict.get("model", None)
            if model_state is None:
                raise KeyError("Missing key 'model' in checkpoint['state_dict']")
            self.model.load_state_dict(model_state, strict=True)
            self._set_eval_mode()
            print(f"[GatingEvaluator] Loaded checkpoint at startup: {ckpt_path}")
        except Exception as e:
            print(f"[GatingEvaluator] Failed to load startup checkpoint: {e}")

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict)}")

        model_state = state_dict.get("model")
        if model_state is None:
            raise KeyError("Missing key 'model' in gating state_dict")

        self.model.load_state_dict(model_state, strict=True)
        self._set_eval_mode()

        print("[GatingEvaluator] Neural weights hot-swapped successfully.")

    @torch.no_grad()
    def evaluate(
        self,
        aggregated_csi_block: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            aggregated_csi_block: [Q, T, N, M]

        Output:
            reliability_qt: [Q, T]
        """
        if not isinstance(aggregated_csi_block, torch.Tensor):
            raise TypeError(
                f"AI Gating fail: aggregated_csi_block must be torch.Tensor, got {type(aggregated_csi_block)}"
            )

        if aggregated_csi_block.device != self.device:
            aggregated_csi_block = aggregated_csi_block.to(self.device, non_blocking=True)

        if aggregated_csi_block.ndim != 4:
            raise ValueError(
                f"AI Gating fail: expected aggregated_csi_block shape [Q, T, N, M], got {tuple(aggregated_csi_block.shape)}"
            )

        num_aps, _, num_antennas, num_subcarriers = aggregated_csi_block.shape

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

        aggregated_csi_batch = aggregated_csi_block.unsqueeze(0)   # [1,Q,T,N,M]
        pattern = self.represent.process(aggregated_csi_batch)     # [1,Q,T,C,M]

        reliability_btq, logits_btq = self.model(
            pattern,
            return_logits=True,
        )

        if reliability_btq.ndim != 3:
            raise RuntimeError(
                f"AI Gating fail: model must output [B,T,Q], got {tuple(reliability_btq.shape)}"
            )

        if logits_btq.ndim != 3:
            raise RuntimeError(
                f"AI Gating fail: logits must have shape [B,T,Q], got {tuple(logits_btq.shape)}"
            )

        reliability_qt = reliability_btq[0].transpose(0, 1).contiguous()  # [Q,T]

        return reliability_qt