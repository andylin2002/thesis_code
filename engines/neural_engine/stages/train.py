# engines/neural_engine/stages/train.py

import torch
import torch.nn.functional as F
import torch.optim as optim

from typing import Any, Dict

from core.models.reliability_model import NeuralReliabilityModel


class TrainStage:
    """Train neural reliability model with eval-aware diagnostics."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.device = device

        self.learning_rate = float(config.get("LR", 1e-3))
        self.grad_clip_norm = float(config.get("GRAD_CLIP_NORM", 1.0))

        self.loss_weight_proxy = float(config.get("LOSS_WEIGHT_PROXY", 1.0))
        self.loss_weight_smooth = float(config.get("LOSS_WEIGHT_SMOOTH", 0.0))

        self.force_eval_mode_during_fit = bool(
            config.get("NEURAL_FORCE_EVAL_MODE_DURING_FIT", False)
        )
        self.freeze_bn_dropout = bool(
            config.get("NEURAL_FREEZE_BN_DROPOUT", False)
        )

        self.model = NeuralReliabilityModel(config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        if self.freeze_bn_dropout:
            self._freeze_bn_and_disable_dropout()

        self.last_debug_tensors: Dict[str, torch.Tensor] = {}

    def _freeze_bn_and_disable_dropout(self) -> None:
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    torch.nn.Dropout,
                    torch.nn.Dropout1d,
                    torch.nn.Dropout2d,
                    torch.nn.Dropout3d,
                ),
            ):
                module.p = 0.0
                module.eval()

            if isinstance(
                module,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                ),
            ):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def _compute_score_loss(
        self,
        pred_logits: torch.Tensor,
        target_score: torch.Tensor,
    ) -> torch.Tensor:
        target_btq = target_score.permute(0, 2, 1).contiguous()
        target_btq = target_btq - target_btq.mean(dim=-1, keepdim=True)
        return F.mse_loss(pred_logits, target_btq)

    def _compute_smoothness_loss(
        self,
        pred_reliability: torch.Tensor,
    ) -> torch.Tensor:
        if pred_reliability.shape[1] < 2:
            return torch.zeros(
                (),
                device=pred_reliability.device,
                dtype=pred_reliability.dtype,
            )

        diff = pred_reliability[:, 1:, :] - pred_reliability[:, :-1, :]
        return (diff ** 2).mean()

    def _target_btq(
        self,
        target_score: torch.Tensor,
    ) -> torch.Tensor:
        return target_score.permute(0, 2, 1).contiguous()

    def _target_centered_btq(
        self,
        target_score: torch.Tensor,
    ) -> torch.Tensor:
        target_btq = self._target_btq(target_score)
        return target_btq - target_btq.mean(dim=-1, keepdim=True)

    def _target_reliability_btq(
        self,
        target_score: torch.Tensor,
    ) -> torch.Tensor:
        target_btq = self._target_btq(target_score)
        q = target_btq.shape[-1]
        return q * torch.softmax(target_btq, dim=-1)

    def _rankdata_1d(self, x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x, stable=True)
        ranks = torch.empty_like(order, dtype=torch.float32)

        i = 0
        n = x.numel()
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]].item() == x[order[i]].item():
                j += 1
            avg_rank = 0.5 * (i + j)
            ranks[order[i : j + 1]] = avg_rank
            i = j + 1

        return ranks

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

                if torch.std(px) < 1e-12 or torch.std(tx) < 1e-12:
                    values.append(0.0)
                    continue

                pr = self._rankdata_1d(px)
                tr = self._rankdata_1d(tx)

                pr = pr - pr.mean()
                tr = tr - tr.mean()

                denom = torch.sqrt((pr ** 2).sum() * (tr ** 2).sum()).item()
                if denom < 1e-12:
                    values.append(0.0)
                    continue

                corr = float((pr * tr).sum().item() / denom)
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

    def _pairwise_acc(
        self,
        pred_btq: torch.Tensor,
        target_btq: torch.Tensor,
    ) -> float:
        q = pred_btq.shape[-1]
        correct = 0
        total = 0

        for i in range(q):
            for j in range(i + 1, q):
                pred_cmp = pred_btq[..., i] > pred_btq[..., j]
                target_cmp = target_btq[..., i] > target_btq[..., j]
                correct += int((pred_cmp == target_cmp).sum().item())
                total += int(pred_cmp.numel())

        if total == 0:
            return 0.0
        return float(correct / total)

    def _rel_stats(
        self,
        reliability_btq: torch.Tensor,
    ) -> Dict[str, float]:
        q = reliability_btq.shape[-1]

        if q >= 2:
            top2_values = torch.topk(reliability_btq, k=2, dim=-1).values
            gap = float((top2_values[..., 0] - top2_values[..., 1]).mean().item())
        else:
            gap = 0.0

        return {
            "pred_mean": float(reliability_btq.mean().item()),
            "pred_max": float(reliability_btq.max().item()),
            "pred_min": float(reliability_btq.min().item()),
            "pred_std": float(reliability_btq.std(dim=-1).mean().item()),
            "top1_top2_gap": gap,
        }

    def _collect_metrics(
        self,
        pred_reliability: torch.Tensor,
        pred_logits: torch.Tensor,
        target_score: torch.Tensor,
        prefix: str,
    ) -> Dict[str, float]:
        target_btq = self._target_btq(target_score)
        target_centered_btq = self._target_centered_btq(target_score)
        target_reliability_btq = self._target_reliability_btq(target_score)

        score_loss = self._compute_score_loss(pred_logits, target_score)

        out = {
            f"{prefix}_score_loss": float(score_loss.item()),
            f"{prefix}_score_spearman": self._mean_spearman(
                pred_logits, target_centered_btq
            ),
            f"{prefix}_score_top1_acc": self._top1_acc(
                pred_logits, target_centered_btq
            ),
            f"{prefix}_reliability_top1_acc": self._top1_acc(
                pred_reliability, target_reliability_btq
            ),
            f"{prefix}_reliability_pairwise_acc": self._pairwise_acc(
                pred_reliability, target_reliability_btq
            ),
        }

        rel_stats = self._rel_stats(pred_reliability)
        for key, value in rel_stats.items():
            out[f"{prefix}_{key}"] = value

        out[f"{prefix}_target_score_mean"] = float(target_btq.mean().item())
        out[f"{prefix}_target_score_max"] = float(target_btq.max().item())
        out[f"{prefix}_target_score_min"] = float(target_btq.min().item())

        return out

    def _cache_debug_tensors(
        self,
        train_pred_reliability: torch.Tensor,
        train_pred_logits: torch.Tensor,
        eval_pred_reliability: torch.Tensor,
        eval_pred_logits: torch.Tensor,
        target_score: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            target_btq = self._target_btq(target_score)
            target_centered_btq = self._target_centered_btq(target_score)

            train_score_mse_map = (train_pred_logits - target_centered_btq) ** 2
            eval_score_mse_map = (eval_pred_logits - target_centered_btq) ** 2

            def _top1_gap(x: torch.Tensor) -> torch.Tensor:
                top2_values, top2_indices = torch.topk(
                    x,
                    k=min(2, x.shape[-1]),
                    dim=-1,
                )

                if x.shape[-1] >= 2:
                    gap = top2_values[..., 0] - top2_values[..., 1]
                else:
                    gap = torch.zeros(
                        x.shape[:2],
                        device=x.device,
                        dtype=x.dtype,
                    )
                return top2_indices[..., 0], gap

            train_top1_ap, train_gap = _top1_gap(train_pred_reliability)
            eval_top1_ap, eval_gap = _top1_gap(eval_pred_reliability)

            self.last_debug_tensors = {
                "train_pred_reliability": train_pred_reliability.detach().cpu(),
                "train_pred_logits": train_pred_logits.detach().cpu(),
                "eval_pred_reliability": eval_pred_reliability.detach().cpu(),
                "eval_pred_logits": eval_pred_logits.detach().cpu(),
                "target_score": target_score.detach().cpu(),
                "target_score_btq": target_btq.detach().cpu(),
                "target_score_centered_btq": target_centered_btq.detach().cpu(),
                "train_score_mse_map": train_score_mse_map.detach().cpu(),
                "eval_score_mse_map": eval_score_mse_map.detach().cpu(),
                "train_top1_ap": train_top1_ap.detach().cpu(),
                "eval_top1_ap": eval_top1_ap.detach().cpu(),
                "train_top1_top2_gap": train_gap.detach().cpu(),
                "eval_top1_top2_gap": eval_gap.detach().cpu(),
            }

    def get_debug_tensors(self) -> Dict[str, torch.Tensor]:
        return self.last_debug_tensors

    def step(
        self,
        pattern: torch.Tensor,
        target_score: torch.Tensor,
    ) -> Dict[str, float]:
        if pattern.ndim != 5:
            raise ValueError(
                f"Expected pattern with shape [B, Q, T, C, M], got {tuple(pattern.shape)}"
            )

        if target_score.ndim != 3:
            raise ValueError(
                f"Expected target_score with shape [B, Q, T], got {tuple(target_score.shape)}"
            )

        if self.force_eval_mode_during_fit:
            self.model.eval()
        else:
            self.model.train()

        train_pred_reliability, train_pred_logits = self.model(
            pattern,
            return_logits=True,
        )

        score_loss = self._compute_score_loss(train_pred_logits, target_score)
        smoothness_loss = self._compute_smoothness_loss(train_pred_reliability)

        total_loss = (
            self.loss_weight_proxy * score_loss
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

        self.model.eval()
        with torch.no_grad():
            eval_pred_reliability, eval_pred_logits = self.model(
                pattern,
                return_logits=True,
            )

        self._cache_debug_tensors(
            train_pred_reliability=train_pred_reliability,
            train_pred_logits=train_pred_logits,
            eval_pred_reliability=eval_pred_reliability,
            eval_pred_logits=eval_pred_logits,
            target_score=target_score,
        )

        metrics = {
            "loss": float(total_loss.item()),
            "score_loss": float(score_loss.item()),
            "smoothness_loss": float(smoothness_loss.item()),
        }

        metrics.update(
            self._collect_metrics(
                pred_reliability=train_pred_reliability.detach(),
                pred_logits=train_pred_logits.detach(),
                target_score=target_score.detach(),
                prefix="train",
            )
        )

        metrics.update(
            self._collect_metrics(
                pred_reliability=eval_pred_reliability.detach(),
                pred_logits=eval_pred_logits.detach(),
                target_score=target_score.detach(),
                prefix="eval",
            )
        )

        # backward-compatible keys for runtime publish gate
        metrics["pred_mean"] = metrics["eval_pred_mean"]
        metrics["pred_max"] = metrics["eval_pred_max"]
        metrics["pred_min"] = metrics["eval_pred_min"]
        metrics["pred_std"] = metrics["eval_pred_std"]
        metrics["target_score_mean"] = metrics["eval_target_score_mean"]
        metrics["target_score_max"] = metrics["eval_target_score_max"]
        metrics["target_score_min"] = metrics["eval_target_score_min"]
        metrics["top1_top2_gap"] = metrics["eval_top1_top2_gap"]

        return metrics

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
        }