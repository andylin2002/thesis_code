# engines/neural_engine/stages/train.py

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim

from core.models.reliability_model import NeuralReliabilityModel


class TrainStage:
    """Train neural model to predict long-horizon AP score."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.config = config
        self.device = device

        self.learning_rate = float(config.get("LR", 1e-3))
        self.weight_decay = float(config.get("WEIGHT_DECAY", 0.0))
        self.grad_clip_norm = float(config.get("GRAD_CLIP_NORM", 1.0))

        self.loss_weight_score = float(config.get("LOSS_WEIGHT_SCORE", 1.0))
        self.loss_weight_smooth = float(config.get("LOSS_WEIGHT_SMOOTH", 0.0))

        self.score_loss_type = str(config.get("SCORE_LOSS_TYPE", "huber")).lower()
        self.huber_beta = float(config.get("HUBER_BETA", 1.0))

        self.model = NeuralReliabilityModel(config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.last_debug_tensors: Dict[str, torch.Tensor] = {}

    def _target_btq(self, target_score: torch.Tensor) -> torch.Tensor:
        if target_score.ndim != 3:
            raise ValueError(
                f"Expected target_score with shape [B,Q,T], got {tuple(target_score.shape)}"
            )
        return target_score.permute(0, 2, 1).contiguous()

    def _target_centered_btq(self, target_score: torch.Tensor) -> torch.Tensor:
        target_btq = self._target_btq(target_score)
        return target_btq - target_btq.mean(dim=-1, keepdim=True)

    def _compute_score_loss(
        self,
        pred_logits: torch.Tensor,
        target_score: torch.Tensor,
    ) -> torch.Tensor:
        target_centered_btq = self._target_centered_btq(target_score)

        if self.score_loss_type == "mse":
            return F.mse_loss(pred_logits, target_centered_btq)

        if self.score_loss_type == "huber":
            return F.smooth_l1_loss(
                pred_logits,
                target_centered_btq,
                beta=self.huber_beta,
            )

        raise ValueError(f"Unsupported SCORE_LOSS_TYPE: {self.score_loss_type}")

    def _compute_smoothness_loss(self, pred_logits: torch.Tensor) -> torch.Tensor:
        if pred_logits.shape[1] < 2:
            return torch.zeros((), device=pred_logits.device, dtype=pred_logits.dtype)

        diff = pred_logits[:, 1:, :] - pred_logits[:, :-1, :]
        return (diff ** 2).mean()

    def _mean_spearman(
        self,
        pred_btq: torch.Tensor,
        target_btq: torch.Tensor,
    ) -> float:
        values = []

        for b in range(pred_btq.shape[0]):
            for t in range(pred_btq.shape[1]):
                px = pred_btq[b, t]
                tx = target_btq[b, t]

                px_rank = torch.argsort(torch.argsort(px))
                tx_rank = torch.argsort(torch.argsort(tx))

                px_rank = px_rank.float()
                tx_rank = tx_rank.float()

                px_rank = px_rank - px_rank.mean()
                tx_rank = tx_rank - tx_rank.mean()

                denom = torch.sqrt((px_rank ** 2).sum() * (tx_rank ** 2).sum()).item()
                if denom < 1e-12:
                    values.append(0.0)
                    continue

                corr = float((px_rank * tx_rank).sum().item() / denom)
                values.append(corr)

        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _top1_acc(
        self,
        pred_btq: torch.Tensor,
        target_btq: torch.Tensor,
    ) -> float:
        pred_top1 = pred_btq.argmax(dim=-1)
        target_top1 = target_btq.argmax(dim=-1)
        return float((pred_top1 == target_top1).float().mean().item())

    def get_debug_tensors(self) -> Dict[str, torch.Tensor]:
        return self.last_debug_tensors

    def step(
        self,
        pattern: torch.Tensor,
        target_score: torch.Tensor,
    ) -> Dict[str, float]:
        if pattern.ndim != 5:
            raise ValueError(
                f"Expected pattern with shape [B,Q,T,C,M], got {tuple(pattern.shape)}"
            )

        if target_score.ndim != 3:
            raise ValueError(
                f"Expected target_score with shape [B,Q,T], got {tuple(target_score.shape)}"
            )

        if pattern.device != self.device:
            pattern = pattern.to(self.device, non_blocking=True)

        if target_score.device != self.device:
            target_score = target_score.to(self.device, non_blocking=True)

        self.model.train()
        _, pred_logits = self.model(pattern, return_logits=True)

        score_loss = self._compute_score_loss(pred_logits, target_score)
        smoothness_loss = self._compute_smoothness_loss(pred_logits)

        total_loss = (
            self.loss_weight_score * score_loss
            + self.loss_weight_smooth * smoothness_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm,
            )

        self.optimizer.step()

        with torch.no_grad():
            target_centered_btq = self._target_centered_btq(target_score)
            pred_centered_btq = pred_logits.detach()

            pred_std = float(pred_centered_btq.std(dim=-1).mean().item())

            if pred_centered_btq.shape[-1] >= 2:
                top2 = torch.topk(pred_centered_btq, k=2, dim=-1).values
                top1_top2_gap = float((top2[..., 0] - top2[..., 1]).mean().item())
            else:
                top1_top2_gap = 0.0

            score_spearman = self._mean_spearman(pred_centered_btq, target_centered_btq)
            score_top1_acc = self._top1_acc(pred_centered_btq, target_centered_btq)

            self.last_debug_tensors = {
                "pred_logits": pred_centered_btq.detach().cpu(),
            }

        return {
            "loss": float(total_loss.item()),
            "score_loss": float(score_loss.item()),
            "smoothness_loss": float(smoothness_loss.item()),
            "score_spearman": score_spearman,
            "score_top1_acc": score_top1_acc,
            "pred_std": pred_std,
            "top1_top2_gap": top1_top2_gap,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
        }