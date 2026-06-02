# engines/neural_engine/stages/train.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
from core.models.model import NeuralReliabilityModel


class TrainStage:
    """Train neural reliability model with richer debug info for train/eval comparison."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.config = config
        self.device = device

        self.learning_rate = float(config.get("LEARNING_RATE", 1e-3))
        self.weight_decay = float(config.get("WEIGHT_DECAY", 0.0))
        self.gradient_clip_norm = float(config.get("GRADIENT_CLIP_NORM", 1.0))

        self.teacher_temperature = float(config.get("TEACHER_TEMPERATURE", 1.0))
        self.student_temperature = float(config.get("STUDENT_TEMPERATURE", 1.0))
        self.teacher_std_floor = float(config.get("TEACHER_STD_FLOOR", 1e-4))
        self.eps = 1e-8

        utils.apply_reproducibility_config(config)
        self.model = NeuralReliabilityModel(config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.last_debug_tensors: Dict[str, torch.Tensor] = {}
        self.last_debug_scalars: Dict[str, float] = {}

    def _target_btq(self, target_score: torch.Tensor) -> torch.Tensor:
        if target_score.ndim != 3:
            raise ValueError(
                f"Expected target_score with shape [B,Q,T], got {tuple(target_score.shape)}"
            )
        return target_score.permute(0, 2, 1).contiguous()

    def _build_teacher_logits_btq(self, target_score: torch.Tensor) -> torch.Tensor:
        target_btq = self._target_btq(target_score)
        target_btq = target_btq - target_btq.mean(dim=-1, keepdim=True)

        std = target_btq.std(dim=-1, keepdim=True).clamp_min(self.teacher_std_floor)
        target_btq = target_btq / std

        return target_btq

    def _build_teacher_prob_btq(
        self,
        target_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        teacher_logits = self._build_teacher_logits_btq(target_score)
        teacher_prob = torch.softmax(
            teacher_logits / self.teacher_temperature,
            dim=-1,
        )
        return teacher_logits, teacher_prob

    def _build_student_log_prob_btq(self, pred_logits: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(pred_logits / self.student_temperature, dim=-1)

    def _compute_proxy_loss(
        self,
        pred_logits: torch.Tensor,
        target_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_logits, teacher_prob = self._build_teacher_prob_btq(target_score)
        student_log_prob = self._build_student_log_prob_btq(pred_logits)

        loss = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="batchmean",
            log_target=False,
        )
        return loss, teacher_logits, teacher_prob

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

                px_rank = torch.argsort(torch.argsort(px)).float()
                tx_rank = torch.argsort(torch.argsort(tx)).float()

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

    def _mean_pearson(
        self,
        pred_btq: torch.Tensor,
        target_btq: torch.Tensor,
    ) -> float:
        values = []

        for b in range(pred_btq.shape[0]):
            for t in range(pred_btq.shape[1]):
                px = pred_btq[b, t] - pred_btq[b, t].mean()
                tx = target_btq[b, t] - target_btq[b, t].mean()

                denom = torch.sqrt((px ** 2).sum() * (tx ** 2).sum()).item()
                if denom < 1e-12:
                    values.append(0.0)
                    continue

                corr = float((px * tx).sum().item() / denom)
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

    def _mean_entropy(self, prob_btq: torch.Tensor) -> float:
        entropy = -(prob_btq.clamp_min(self.eps) * prob_btq.clamp_min(self.eps).log()).sum(dim=-1)
        return float(entropy.mean().item())

    def _mean_max_prob(self, prob_btq: torch.Tensor) -> float:
        return float(prob_btq.max(dim=-1).values.mean().item())

    def _top1_top2_gap(self, logits_btq: torch.Tensor) -> float:
        if logits_btq.shape[-1] < 2:
            return 0.0
        top2 = torch.topk(logits_btq, k=2, dim=-1).values
        return float((top2[..., 0] - top2[..., 1]).mean().item())

    def _collect_stats(
        self,
        pred_logits_btq: torch.Tensor,
        pred_prob_btq: torch.Tensor,
        teacher_logits_btq: torch.Tensor,
        teacher_prob_btq: torch.Tensor,
        prefix: str,
    ) -> Dict[str, float]:
        return {
            f"{prefix}_logits_std": float(pred_logits_btq.std(dim=-1).mean().item()),
            f"{prefix}_logits_min": float(pred_logits_btq.min().item()),
            f"{prefix}_logits_max": float(pred_logits_btq.max().item()),
            f"{prefix}_gap_logit": self._top1_top2_gap(pred_logits_btq),
            f"{prefix}_entropy": self._mean_entropy(pred_prob_btq),
            f"{prefix}_max_prob": self._mean_max_prob(pred_prob_btq),
            f"{prefix}_top1_acc_prob": self._top1_acc(pred_prob_btq, teacher_prob_btq),
            f"{prefix}_top1_acc_logit": self._top1_acc(pred_logits_btq, teacher_logits_btq),
            f"{prefix}_mean_spearman_logit": self._mean_spearman(pred_logits_btq, teacher_logits_btq),
            f"{prefix}_mean_pearson_logit": self._mean_pearson(pred_logits_btq, teacher_logits_btq),
        }

    def get_debug_tensors(self) -> Dict[str, torch.Tensor]:
        return self.last_debug_tensors

    def get_debug_scalars(self) -> Dict[str, float]:
        return self.last_debug_scalars

    def step(
        self,
        pattern: torch.Tensor,
        target_score: torch.Tensor
    ) -> Dict[str, float]:
        if pattern.ndim != 5:
            raise ValueError(
                f"Expected pattern shape [B,Q,T,C,M], got {tuple(pattern.shape)}"
            )

        if target_score.ndim != 3:
            raise ValueError(
                f"Expected target_score shape [B,Q,T], got {tuple(target_score.shape)}"
            )

        if pattern.device != self.device:
            pattern = pattern.to(self.device, non_blocking=True)

        if target_score.device != self.device:
            target_score = target_score.to(self.device, non_blocking=True)

        teacher_logits_btq, teacher_prob_btq = self._build_teacher_prob_btq(target_score)

        self.model.train()
        _, train_pred_logits_btq, train_embedding_be = self.model(
            pattern,
            return_logits=True,
            return_embedding=True,
        )

        proxy_loss = F.kl_div(
            self._build_student_log_prob_btq(train_pred_logits_btq),
            teacher_prob_btq,
            reduction="batchmean",
            log_target=False,
        )

        total_loss = proxy_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm,
            )

        self.optimizer.step()

        with torch.no_grad():
            train_pred_logits_btq = train_pred_logits_btq.detach()
            train_pred_prob_btq = torch.softmax(train_pred_logits_btq, dim=-1)
            train_embedding_be = train_embedding_be.detach()

            self.model.eval()
            _, eval_pred_logits_btq, eval_embedding_be = self.model(
                pattern,
                return_logits=True,
                return_embedding=True,
            )
            eval_pred_logits_btq = eval_pred_logits_btq.detach()
            eval_pred_prob_btq = torch.softmax(eval_pred_logits_btq, dim=-1)
            eval_embedding_be = eval_embedding_be.detach()

            self.model.train()

            teacher_gap = self._top1_top2_gap(teacher_logits_btq)
            teacher_entropy = self._mean_entropy(teacher_prob_btq)
            teacher_max_prob = self._mean_max_prob(teacher_prob_btq)

            train_eval_logit_abs_diff = float(
                (train_pred_logits_btq - eval_pred_logits_btq).abs().mean().item()
            )
            train_eval_prob_abs_diff = float(
                (train_pred_prob_btq - eval_pred_prob_btq).abs().mean().item()
            )
            train_eval_top1_consistency = float(
                (
                    train_pred_logits_btq.argmax(dim=-1)
                    == eval_pred_logits_btq.argmax(dim=-1)
                ).float().mean().item()
            )
            embedding_abs_diff = float(
                (train_embedding_be - eval_embedding_be).abs().mean().item()
            )

            emb_norm_train = float(train_embedding_be.norm(dim=-1).mean().item())
            emb_norm_eval = float(eval_embedding_be.norm(dim=-1).mean().item())

            metrics = {
                "loss": float(total_loss.item()),
                "proxy_loss": float(proxy_loss.item()),
                "teacher_gap_logit": teacher_gap,
                "teacher_entropy": teacher_entropy,
                "teacher_max_prob": teacher_max_prob,
                "embedding_norm_train": emb_norm_train,
                "embedding_norm_eval": emb_norm_eval,
                "train_eval_logit_abs_diff": train_eval_logit_abs_diff,
                "train_eval_prob_abs_diff": train_eval_prob_abs_diff,
                "train_eval_top1_consistency": train_eval_top1_consistency,
                "embedding_abs_diff": embedding_abs_diff,
            }

            metrics.update(
                self._collect_stats(
                    train_pred_logits_btq,
                    train_pred_prob_btq,
                    teacher_logits_btq,
                    teacher_prob_btq,
                    prefix="train",
                )
            )
            metrics.update(
                self._collect_stats(
                    eval_pred_logits_btq,
                    eval_pred_prob_btq,
                    teacher_logits_btq,
                    teacher_prob_btq,
                    prefix="eval",
                )
            )

            self.last_debug_tensors = {
                "pattern": pattern.detach().cpu(),
                "target_score": target_score.detach().cpu(),
                "teacher_logits": teacher_logits_btq.detach().cpu(),
                "teacher_prob": teacher_prob_btq.detach().cpu(),
                "train_pred_logits": train_pred_logits_btq.cpu(),
                "train_pred_prob": train_pred_prob_btq.cpu(),
                "eval_pred_logits": eval_pred_logits_btq.cpu(),
                "eval_pred_prob": eval_pred_prob_btq.cpu(),
                "train_embedding": train_embedding_be.cpu(),
                "eval_embedding": eval_embedding_be.cpu(),
            }
            self.last_debug_scalars = metrics

        return metrics

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
        }