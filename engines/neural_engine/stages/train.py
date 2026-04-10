# engines/neural_engine/stages/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Any, Dict, Tuple


class CrossAPReliabilityHead(nn.Module):
    """
    Map encoded AP features [B, T, Q, D] to:
        logits: [B, T, Q]
        reliability: [B, T, Q], with sum_q reliability = Q
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(feature_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        encoded: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if encoded.ndim != 4:
            raise ValueError(
                f"Expected encoded with shape [B, T, Q, D], got {tuple(encoded.shape)}"
            )

        _, _, num_aps, _ = encoded.shape

        x = self.norm(encoded)                    # [B, T, Q, D]
        logits = self.score_mlp(x).squeeze(-1)   # [B, T, Q]
        reliability = num_aps * F.softmax(logits, dim=-1)

        if return_logits:
            return reliability, logits
        return reliability


class TrainStage:
    """
    First-version training stage for AP reliability learning.

    Loss:
        total = lambda_consistency * L_consistency
              + lambda_ranking * L_ranking
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        encoder: nn.Module,
    ):
        self.device = device
        self.encoder = encoder

        self.latent_dim = int(config.get("LATENT_DIM", 128))
        self.learning_rate = float(config.get("LR", 5e-4))
        self.grad_clip_norm = float(config.get("GRAD_CLIP_NORM", 1.0))

        # Stronger ranking by default
        self.lambda_consistency = float(config.get("LOSS_WEIGHT_CONS", 1.0))
        self.lambda_ranking = float(config.get("LOSS_WEIGHT_RANK", 1.0))

        # Mild augmentations
        self.subcarrier_keep_ratio = float(config.get("SUBCARRIER_KEEP_RATIO", 0.85))
        self.feature_dropout_prob = float(config.get("FEATURE_DROPOUT_PROB", 0.05))
        self.amplitude_jitter_std = float(config.get("AMPLITUDE_JITTER_STD", 0.02))
        self.phase_jitter_std = float(config.get("PHASE_JITTER_STD", 0.01))

        self.rank_margin = float(config.get("RANK_MARGIN", 0.05))
        self.rank_gap_threshold = float(config.get("RANK_GAP_THRESHOLD", 0.05))

        # Proxy weights
        self.proxy_alpha = float(config.get("PROXY_ALPHA", 1.0))   # phase variance
        self.proxy_beta = float(config.get("PROXY_BETA", 0.5))     # |skewness|
        self.proxy_gamma = float(config.get("PROXY_GAMMA", 0.0))   # |excess kurtosis|

        self.eps = 1e-8

        self.reliability_head = CrossAPReliabilityHead(
            feature_dim=self.latent_dim,
            hidden_dim=int(config.get("RELIABILITY_HEAD_HIDDEN", 64)),
            dropout=float(config.get("RELIABILITY_HEAD_DROPOUT", 0.1)),
        ).to(self.device)

        self.optimizer = optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.reliability_head.parameters()},
            ],
            lr=self.learning_rate,
        )

        self.last_debug_tensors: Dict[str, torch.Tensor] = {}

    # --------------------------------------------------
    # Feature-level augmentation
    # --------------------------------------------------
    def _random_subcarrier_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Q, T, C, M]
        Keep a random subset of subcarriers.
        """
        if self.subcarrier_keep_ratio >= 1.0:
            return x

        bsz, num_aps, num_steps, _, num_subcarriers = x.shape
        keep = max(1, int(round(num_subcarriers * self.subcarrier_keep_ratio)))

        mask = torch.zeros(
            (bsz, num_aps, num_steps, 1, num_subcarriers),
            device=x.device,
            dtype=x.dtype,
        )

        for b in range(bsz):
            for q in range(num_aps):
                for t in range(num_steps):
                    idx = torch.randperm(num_subcarriers, device=x.device)[:keep]
                    mask[b, q, t, 0, idx] = 1.0

        return x * mask

    def _feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_dropout_prob <= 0.0:
            return x
        return F.dropout(x, p=self.feature_dropout_prob, training=True)

    def _add_small_jitter(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()

        # amplitude
        x[:, :, :, 0:1, :] = (
            x[:, :, :, 0:1, :]
            + torch.randn_like(x[:, :, :, 0:1, :]) * self.amplitude_jitter_std
        )

        # phase difference
        if x.shape[3] > 1:
            x[:, :, :, 1:, :] = (
                x[:, :, :, 1:, :]
                + torch.randn_like(x[:, :, :, 1:, :]) * self.phase_jitter_std
            )

        return x

    def _build_two_views(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        view_a = self._random_subcarrier_mask(input_features)
        view_b = self._random_subcarrier_mask(input_features)

        view_a = self._feature_dropout(view_a)
        view_b = self._feature_dropout(view_b)

        view_a = self._add_small_jitter(view_a)
        view_b = self._add_small_jitter(view_b)

        return view_a, view_b

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------
    def _encode_from_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            input_features: [B, Q, T, C, M]

        Output:
            encoded: [B, T, Q, D]
        """
        if input_features.ndim != 5:
            raise ValueError(
                f"Expected input_features with shape [B, Q, T, C, M], got {tuple(input_features.shape)}"
            )

        batch_size, num_aps, num_steps, num_channels, num_subcarriers = input_features.shape

        encoder_input = input_features.view(
            batch_size * num_aps,
            num_steps,
            num_channels,
            num_subcarriers,
        )  # [B*Q, T, C, M]

        encoded_per_ap = self.encoder(
            encoder_input,
            return_projection=False,
        )  # [B*Q, T, D]

        encoded = encoded_per_ap.view(
            batch_size,
            num_aps,
            num_steps,
            -1,
        ).permute(0, 2, 1, 3).contiguous()  # [B, T, Q, D]

        return encoded

    # --------------------------------------------------
    # Loss 1: subset consistency
    # --------------------------------------------------
    def _compute_consistency_loss(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(logits_a, logits_b)

    # --------------------------------------------------
    # Weak proxy statistics
    # --------------------------------------------------
    def _normalize_across_aps(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, Q]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std

    def _compute_phase_variance(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        input_features: [B, Q, T, C, M]
        phase channels = 1..C-1

        Return:
            phase_var: [B, T, Q]
        """
        if input_features.shape[3] <= 1:
            return torch.zeros(
                input_features.shape[0],
                input_features.shape[2],
                input_features.shape[1],
                device=input_features.device,
                dtype=input_features.dtype,
            )

        phase_channels = input_features[:, :, :, 1:, :]                # [B, Q, T, Cp, M]
        phase_var = phase_channels.var(dim=(-1, -2), unbiased=False)   # [B, Q, T]
        return phase_var.permute(0, 2, 1).contiguous()                 # [B, T, Q]

    def _compute_amplitude_skewness(self, input_features: torch.Tensor) -> torch.Tensor:
        amplitude = input_features[:, :, :, 0, :]  # [B, Q, T, M]

        mean = amplitude.mean(dim=-1, keepdim=True)
        std = amplitude.std(dim=-1, keepdim=True).clamp_min(self.eps)

        z = (amplitude - mean) / std
        skewness = (z ** 3).mean(dim=-1)           # [B, Q, T]

        return skewness.permute(0, 2, 1).contiguous()  # [B, T, Q]

    def _compute_amplitude_kurtosis(self, input_features: torch.Tensor) -> torch.Tensor:
        amplitude = input_features[:, :, :, 0, :]  # [B, Q, T, M]

        mean = amplitude.mean(dim=-1, keepdim=True)
        std = amplitude.std(dim=-1, keepdim=True).clamp_min(self.eps)

        z = (amplitude - mean) / std
        kurtosis = (z ** 4).mean(dim=-1)           # [B, Q, T]

        return kurtosis.permute(0, 2, 1).contiguous()  # [B, T, Q]

    def _build_proxy_score(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        More stable weak proxy:
            u = -a * norm(phase_var)
                -b * norm(|skewness|)
                -c * norm(|kurtosis - 3|)

        Output:
            proxy_score: [B, T, Q]
        """
        phase_var = self._compute_phase_variance(input_features)                    # [B, T, Q]
        abs_skewness = torch.abs(self._compute_amplitude_skewness(input_features))  # [B, T, Q]
        abs_excess_kurt = torch.abs(
            self._compute_amplitude_kurtosis(input_features) - 3.0
        )                                                                           # [B, T, Q]

        phase_var = self._normalize_across_aps(phase_var)
        abs_skewness = self._normalize_across_aps(abs_skewness)
        abs_excess_kurt = self._normalize_across_aps(abs_excess_kurt)

        proxy_score = (
            -self.proxy_alpha * phase_var
            -self.proxy_beta * abs_skewness
            -self.proxy_gamma * abs_excess_kurt
        )

        return proxy_score

    # --------------------------------------------------
    # Loss 2: weak ranking
    # --------------------------------------------------
    def _compute_pairwise_ranking_loss(
        self,
        logits: torch.Tensor,
        proxy_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits: [B, T, Q]
        proxy_score: [B, T, Q]

        Pairwise logistic ranking loss.
        Ignore pairs whose proxy gap is too small.
        """
        num_aps = logits.shape[-1]

        logit_gap = logits.unsqueeze(-1) - logits.unsqueeze(-2)          # [B, T, Q, Q]
        proxy_gap = proxy_score.unsqueeze(-1) - proxy_score.unsqueeze(-2)

        valid_mask = proxy_gap > self.rank_gap_threshold

        eye = torch.eye(num_aps, device=logits.device, dtype=torch.bool)
        valid_mask = valid_mask & (~eye.unsqueeze(0).unsqueeze(0))

        pair_loss = F.softplus(-(logit_gap - self.rank_margin))

        valid_count = valid_mask.sum()
        if valid_count.item() == 0:
            zero = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            return zero, zero

        ranking_loss = pair_loss[valid_mask].mean()
        valid_ratio = valid_mask.float().mean()

        return ranking_loss, valid_ratio

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------
    def _cache_debug_tensors(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        reliability: torch.Tensor,
        proxy_score: torch.Tensor,
        phase_var: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            top2_values, top2_indices = torch.topk(
                reliability,
                k=min(2, reliability.shape[-1]),
                dim=-1,
            )

            if reliability.shape[-1] >= 2:
                top1_top2_gap = top2_values[..., 0] - top2_values[..., 1]
            else:
                top1_top2_gap = torch.zeros(
                    reliability.shape[:2],
                    device=reliability.device,
                    dtype=reliability.dtype,
                )

            self.last_debug_tensors = {
                "logits_a": logits_a.detach().cpu(),
                "logits_b": logits_b.detach().cpu(),
                "reliability": reliability.detach().cpu(),
                "proxy_score": proxy_score.detach().cpu(),
                "phase_var": phase_var.detach().cpu(),
                "top1_ap": top2_indices[..., 0].detach().cpu(),
                "top1_top2_gap": top1_top2_gap.detach().cpu(),
            }

    def get_debug_tensors(self) -> Dict[str, torch.Tensor]:
        return self.last_debug_tensors

    # --------------------------------------------------
    # Training step
    # --------------------------------------------------
    def step(self, input_features: torch.Tensor) -> Dict[str, float]:
        """
        Input:
            input_features: [B, Q, T, C, M]

        Returns:
            training metrics dict
        """
        if input_features.ndim != 5:
            raise ValueError(
                f"Expected input_features with shape [B, Q, T, C, M], got {tuple(input_features.shape)}"
            )

        self.encoder.train()
        self.reliability_head.train()

        # 1) two mild views
        view_a, view_b = self._build_two_views(input_features)

        # 2) encode
        encoded_a = self._encode_from_features(view_a)  # [B, T, Q, D]
        encoded_b = self._encode_from_features(view_b)  # [B, T, Q, D]

        # 3) score
        reliability_a, logits_a = self.reliability_head(encoded_a, return_logits=True)
        reliability_b, logits_b = self.reliability_head(encoded_b, return_logits=True)

        # 4) consistency
        consistency_loss = self._compute_consistency_loss(logits_a, logits_b)

        # 5) weak ranking
        with torch.no_grad():
            proxy_score = self._build_proxy_score(input_features)       # [B, T, Q]
            phase_var = self._compute_phase_variance(input_features)    # [B, T, Q]

        ranking_loss, valid_pair_ratio = self._compute_pairwise_ranking_loss(
            logits=0.5 * (logits_a + logits_b),
            proxy_score=proxy_score,
        )

        total_loss = (
            self.lambda_consistency * consistency_loss
            + self.lambda_ranking * ranking_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.reliability_head.parameters()),
                max_norm=self.grad_clip_norm,
            )

        self.optimizer.step()

        reliability = 0.5 * (reliability_a + reliability_b)

        self._cache_debug_tensors(
            logits_a=logits_a,
            logits_b=logits_b,
            reliability=reliability,
            proxy_score=proxy_score,
            phase_var=phase_var,
        )

        with torch.no_grad():
            top2_values, _ = torch.topk(
                reliability,
                k=min(2, reliability.shape[-1]),
                dim=-1,
            )

            if reliability.shape[-1] >= 2:
                top1_top2_gap = (top2_values[..., 0] - top2_values[..., 1]).mean()
            else:
                top1_top2_gap = torch.tensor(0.0, device=reliability.device)

        return {
            "loss": total_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "ranking_loss": ranking_loss.item(),
            "valid_pair_ratio": valid_pair_ratio.item(),
            "mean_reliability": reliability.mean().item(),
            "max_reliability": reliability.max().item(),
            "min_reliability": reliability.min().item(),
            "top1_top2_gap": top1_top2_gap.item(),
            "proxy_score_mean": proxy_score.mean().item(),
            "phase_var_mean": phase_var.mean().item(),
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": {k: v.cpu() for k, v in self.encoder.state_dict().items()},
            "reliability_head": {
                k: v.cpu() for k, v in self.reliability_head.state_dict().items()
            },
        }