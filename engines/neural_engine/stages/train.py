# engines/neural_engine/stages/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Dict, Tuple


class TemporalPredictor(nn.Module):
    """
    Predict the next-step latent state z_{t+1} from z_t.

    Input:
        z_t: [B, T-1, Q, D]
    Output:
        z_t_pred: [B, T-1, Q, D]
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.transition(z_t)


class TrainStage:
    """
    Self-supervised training stage for AP-wise relative predictability learning.

    Core idea:
        1. Use temporal prediction error as a weak self-supervised signal.
        2. Convert per-time per-AP relative prediction error into pseudo target probs.
        3. Train the attention head to produce AP-wise relative probabilities.
        4. Encourage the probability distribution to have enough discriminability
           (top1-top2 gap), instead of collapsing toward uniform.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        encoder: nn.Module,
        attention: nn.Module,
    ):
        self.device = device

        self.latent_dim = int(config.get("LATENT_DIM", 128))
        self.lr = float(config.get("LR", 5e-4))

        self.prob_target_lambda = float(config.get("PROB_TARGET_LAMBDA", 1.0))
        self.var_reg_lambda = float(config.get("VAR_REG_LAMBDA", 0.02))
        self.target_temperature = float(config.get("TARGET_TEMPERATURE", 0.5))

        # 保留欄位相容性，但這個版本不再使用 smoothing
        self.label_smoothing = float(config.get("LABEL_SMOOTHING", 0.0))

        self.eps = 1e-8

        self.encoder = encoder
        self.attention = attention
        self.predictor = TemporalPredictor(self.latent_dim).to(self.device)

        self.optimizer = optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.attention.parameters()},
                {"params": self.predictor.parameters()},
            ],
            lr=self.lr,
        )

        # Cache the latest debug tensors for runtime export
        self.last_debug_tensors: Dict[str, torch.Tensor] = {}

    def _build_probability_target(self, pred_error: torch.Tensor) -> torch.Tensor:
        """
        Convert per-AP prediction errors into pseudo probability targets.

        IMPORTANT:
            We do relative comparison across APs at each (B, t), instead of using raw
            error scale directly. This makes the target more aligned with the task:
            "which AP is more predictable / stable than the others at this time?"

        Args:
            pred_error: [B, T-1, Q]

        Returns:
            target_probs: [B, T-1, Q], sum over Q = 1
        """
        # Normalize error across APs at each (B,t)
        err_mean = pred_error.mean(dim=-1, keepdim=True)                     # [B,T-1,1]
        err_std = pred_error.std(dim=-1, keepdim=True).clamp_min(1e-6)      # [B,T-1,1]
        pred_error_norm = (pred_error - err_mean) / err_std                 # [B,T-1,Q]

        # Lower error -> larger probability
        target_probs = F.softmax(-pred_error_norm / self.target_temperature, dim=-1)

        # Intentionally DO NOT apply label smoothing here.
        # We want to preserve relative AP discriminability.
        return target_probs

    def _cache_debug_tensors(
        self,
        pred_error: torch.Tensor,
        target_probs: torch.Tensor,
        probs_t: torch.Tensor,
        scores_t: torch.Tensor,
    ) -> None:
        """
        Save detached CPU copies for later analysis / npy export.
        """
        with torch.no_grad():
            top2_vals, top2_idx = torch.topk(
                probs_t, k=min(2, probs_t.shape[-1]), dim=-1
            )

            if probs_t.shape[-1] >= 2:
                prob_gap = top2_vals[..., 0] - top2_vals[..., 1]   # [B, T-1]
            else:
                prob_gap = torch.zeros(
                    probs_t.shape[:2], device=probs_t.device, dtype=probs_t.dtype
                )

            top1_ap = top2_idx[..., 0]                             # [B, T-1]
            score_std = scores_t.std(dim=-1)                      # [B, T-1]

            self.last_debug_tensors = {
                "pred_error": pred_error.detach().cpu(),      # [B, T-1, Q]
                "target_probs": target_probs.detach().cpu(),  # [B, T-1, Q]
                "ap_probs": probs_t.detach().cpu(),           # [B, T-1, Q]
                "scores": scores_t.detach().cpu(),            # [B, T-1, Q]
                "score_std": score_std.detach().cpu(),        # [B, T-1]
                "prob_gap": prob_gap.detach().cpu(),          # [B, T-1]
                "top1_ap": top1_ap.detach().cpu(),            # [B, T-1]
            }

    def get_debug_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Return the latest cached debug tensors.
        """
        return self.last_debug_tensors

    def _compute_losses(
        self,
        pred_error: torch.Tensor,
        probs_t: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the three core loss terms.

        Returns:
            pred_loss
            prob_target_loss
            gap_reg_loss
            total_loss
        """
        # Weighted predictive loss:
        # put higher probability on APs that are more predictable
        pred_loss = (pred_error * probs_t).sum(dim=-1).mean()

        # Distribution matching:
        # use KL instead of MSE for probability distributions
        log_probs_t = torch.log(probs_t.clamp_min(self.eps))
        prob_target_loss = F.kl_div(
            log_probs_t,
            target_probs.detach(),
            reduction="batchmean"
        )

        # Directly encourage discriminability:
        # larger top1-top2 gap => less uniform / more decisive ranking
        if probs_t.shape[-1] >= 2:
            top2_vals, _ = torch.topk(probs_t, k=2, dim=-1)
            gap = top2_vals[..., 0] - top2_vals[..., 1]   # [B, T-1]
            gap_reg_loss = -gap.mean()
        else:
            gap_reg_loss = torch.tensor(0.0, device=probs_t.device)

        total_loss = (
            pred_loss
            + self.prob_target_lambda * prob_target_loss
            + self.var_reg_lambda * gap_reg_loss
        )

        return pred_loss, prob_target_loss, gap_reg_loss, total_loss

    def step(self, z_features: torch.Tensor, ap_probs: torch.Tensor) -> Dict[str, float]:
        """
        One self-supervised training step.

        Args:
            z_features: [B, T, Q, D]
            ap_probs:   [B, T, Q], currently unused directly for training loss

        Returns:
            Dictionary of scalar logs.
        """
        self.encoder.train()
        self.attention.train()
        self.predictor.train()

        # ------------------------------------------------------------
        # 1. Prepare one-step temporal prediction pairs
        # ------------------------------------------------------------
        z_t = z_features[:, :-1, :, :]         # [B, T-1, Q, D]
        z_next = z_features[:, 1:, :, :]       # [B, T-1, Q, D]

        # IMPORTANT:
        # Recompute attention probabilities on z_t using the updated attention module.
        # Do not rely on externally passed ap_probs slicing, because we now want raw scores too.
        probs_t, scores_t = self.attention(z_t, return_scores=True)   # [B,T-1,Q], [B,T-1,Q]

        # ------------------------------------------------------------
        # 2. Predict next latent state
        # ------------------------------------------------------------
        z_pred = self.predictor(z_t)           # [B, T-1, Q, D]

        # ------------------------------------------------------------
        # 3. Per-AP prediction error
        # ------------------------------------------------------------
        pred_error = F.mse_loss(
            z_pred,
            z_next.detach(),
            reduction="none"
        ).mean(dim=-1)                         # [B, T-1, Q]

        # ------------------------------------------------------------
        # 4. Build pseudo probability targets
        # ------------------------------------------------------------
        with torch.no_grad():
            target_probs = self._build_probability_target(pred_error)  # [B, T-1, Q]

        # ------------------------------------------------------------
        # 5. Loss terms
        # ------------------------------------------------------------
        pred_loss, prob_target_loss, gap_reg_loss, total_loss = self._compute_losses(
            pred_error=pred_error,
            probs_t=probs_t,
            target_probs=target_probs,
        )

        # ------------------------------------------------------------
        # 6. Backprop
        # ------------------------------------------------------------
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # ------------------------------------------------------------
        # 7. Cache debug tensors
        # ------------------------------------------------------------
        self._cache_debug_tensors(
            pred_error=pred_error,
            target_probs=target_probs,
            probs_t=probs_t,
            scores_t=scores_t,
        )

        # ------------------------------------------------------------
        # 8. Monitoring
        # ------------------------------------------------------------
        with torch.no_grad():
            prob_entropy = -torch.sum(
                probs_t * torch.log(probs_t.clamp_min(self.eps)), dim=-1
            ).mean()

            target_entropy = -torch.sum(
                target_probs * torch.log(target_probs.clamp_min(self.eps)), dim=-1
            ).mean()

            top2_vals, _ = torch.topk(probs_t, k=min(2, probs_t.shape[-1]), dim=-1)
            if probs_t.shape[-1] >= 2:
                top1_top2_gap = (top2_vals[..., 0] - top2_vals[..., 1]).mean()
            else:
                top1_top2_gap = torch.tensor(0.0, device=probs_t.device)

            score_std_mean = scores_t.std(dim=-1).mean()

            prob_variance = probs_t.var(dim=-1).mean()

        return {
            "loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "prob_target_loss": prob_target_loss.item(),
            "gap_reg_loss": gap_reg_loss.item(),
            "prob_variance": prob_variance.item(),
            "prob_entropy": prob_entropy.item(),
            "target_entropy": target_entropy.item(),
            "top1_top2_gap": top1_top2_gap.item(),
            "score_std_mean": score_std_mean.item(),
            "mean_prob": probs_t.mean().item(),
            "max_prob": probs_t.max().item(),
            "min_prob": probs_t.min().item(),
            "mean_pred_error": pred_error.mean().item(),
            "max_pred_error": pred_error.max().item(),
            "min_pred_error": pred_error.min().item(),
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": {k: v.cpu() for k, v in self.encoder.state_dict().items()},
            "attention": {k: v.cpu() for k, v in self.attention.state_dict().items()},
            "predictor": {k: v.cpu() for k, v in self.predictor.state_dict().items()},
        }