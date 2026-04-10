# engines/neural_engine/stages/represent.py

import torch
from typing import Any, Dict, List, Tuple

from core.models.csi_encoder import CSIEncoder


class RepresentStage:
    """
    RepresentStage:
        raw CSI [B, Q, T, N, M] (complex)
            -> build neural features [B, Q, T, C, M]
            -> encode with CSIEncoder
            -> output encoded [B, T, Q, D]
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.num_aps = len(config["ACCESS_POINTS"])
        self.num_rx_antennas = int(config["CSI_DIMENSIONS"]["NUM_RX_ANTENNAS"])
        self.num_subcarriers = int(config["CSI_DIMENSIONS"]["NUM_SUBCARRIERS"])
        self.latent_dim = int(config.get("LATENT_DIM", 128))

        if self.num_rx_antennas < 2:
            raise ValueError("NUM_RX_ANTENNAS must be at least 2")

        self.antenna_pairs = self._build_antenna_pairs(self.num_rx_antennas)
        self.num_phase_diff_channels = len(self.antenna_pairs)
        self.num_feature_channels = 1 + self.num_phase_diff_channels

        self.center_phase_over_subcarriers = bool(
            config.get("NEURAL_CENTER_PHASE_OVER_SUBCARRIERS", True)
        )

        self.encoder = CSIEncoder(
            num_feature_channels=self.num_feature_channels,
            num_subcarriers=self.num_subcarriers,
            cnn_channels=config.get("NEURAL_CNN_CHANNELS", [16, 32]),
            cnn_kernel_size=int(config.get("NEURAL_CNN_KERNEL_SIZE", 3)),
            tcn_hidden=int(config.get("NEURAL_TCN_HIDDEN", 64)),
            tcn_kernel_size=int(config.get("NEURAL_TCN_KERNEL_SIZE", 3)),
            tcn_dilations=config.get("NEURAL_TCN_DILATIONS", [1, 2, 4]),
            mlp_hidden=int(config.get("NEURAL_MLP_HIDDEN", 64)),
            latent_dim=self.latent_dim,
            proj_dim=int(config.get("PROJ_DIM", 64)),
            dropout=float(config.get("NEURAL_DROPOUT", 0.1)),
        ).to(self.device)

    def process(self, raw_csi: torch.Tensor) -> torch.Tensor:
        """
        Input:
            raw_csi: [B, Q, T, N, M] (complex tensor)

        Output:
            encoded: [B, T, Q, D]
        """
        if raw_csi.device != self.device:
            raw_csi = raw_csi.to(self.device, non_blocking=True)

        if raw_csi.ndim != 5:
            raise ValueError(
                f"Expected raw_csi with shape [B, Q, T, N, M], got {tuple(raw_csi.shape)}"
            )

        batch_size, num_aps, num_steps, num_antennas, num_subcarriers = raw_csi.shape

        if num_aps != self.num_aps:
            raise ValueError(f"Configured num_aps={self.num_aps}, but got {num_aps}")
        if num_antennas != self.num_rx_antennas:
            raise ValueError(
                f"Configured num_rx_antennas={self.num_rx_antennas}, but got {num_antennas}"
            )
        if num_subcarriers != self.num_subcarriers:
            raise ValueError(
                f"Configured num_subcarriers={self.num_subcarriers}, but got {num_subcarriers}"
            )

        input_features = self._build_input_features(raw_csi)   # [B, Q, T, C, M]
        encoded = self._encode_features(input_features)        # [B, T, Q, D]
        return encoded

    def _build_antenna_pairs(self, num_antennas: int) -> List[Tuple[int, int]]:
        pairs = []
        for i in range(num_antennas):
            for j in range(i + 1, num_antennas):
                pairs.append((i, j))
        return pairs

    def _build_input_features(self, raw_csi: torch.Tensor) -> torch.Tensor:
        """
        Build neural input features from raw CSI.

        Input:
            raw_csi: [B, Q, T, N, M] (complex)

        Output:
            input_features: [B, Q, T, C, M]
                channel 0: normalized amplitude
                channel 1..: wrapped phase-difference channels in [-1, 1]
        """
        eps = 1e-8

        # ----- amplitude -----
        amplitude = torch.abs(raw_csi)  # [B, Q, T, N, M]

        amp_mean = amplitude.mean(dim=(-1, -2), keepdim=True).clamp_min(eps)
        amplitude_norm = amplitude / amp_mean

        amplitude_channel = amplitude_norm.mean(dim=3)            # [B, Q, T, M]
        amplitude_channel = amplitude_channel.unsqueeze(3)        # [B, Q, T, 1, M]

        amp_sub_mean = amplitude_channel.mean(dim=-1, keepdim=True)
        amp_sub_std = amplitude_channel.std(dim=-1, keepdim=True).clamp_min(eps)
        amplitude_channel = (amplitude_channel - amp_sub_mean) / amp_sub_std

        # ----- phase difference -----
        phase = torch.angle(raw_csi)  # [B, Q, T, N, M]
        phase_diff_channels = []

        for i, j in self.antenna_pairs:
            phase_diff = phase[:, :, :, j, :] - phase[:, :, :, i, :]  # [B, Q, T, M]

            wrapped = torch.atan2(
                torch.sin(phase_diff),
                torch.cos(phase_diff),
            )  # [-pi, pi]

            # Keep true spread information for proxy statistics
            wrapped = wrapped / torch.pi  # [-1, 1]

            if self.center_phase_over_subcarriers:
                wrapped = wrapped - wrapped.mean(dim=-1, keepdim=True)

            phase_diff_channels.append(wrapped.unsqueeze(3))  # [B, Q, T, 1, M]

        phase_diff_tensor = torch.cat(phase_diff_channels, dim=3)  # [B, Q, T, Cp, M]

        input_features = torch.cat(
            [amplitude_channel, phase_diff_tensor],
            dim=3,
        )  # [B, Q, T, C, M]

        return input_features

    def _encode_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            input_features: [B, Q, T, C, M]

        Output:
            encoded: [B, T, Q, D]
        """
        batch_size, num_aps, num_steps, num_channels, num_subcarriers = input_features.shape

        if num_channels != self.num_feature_channels:
            raise ValueError(
                f"Expected num_feature_channels={self.num_feature_channels}, got {num_channels}"
            )
        if num_subcarriers != self.num_subcarriers:
            raise ValueError(
                f"Expected num_subcarriers={self.num_subcarriers}, got {num_subcarriers}"
            )

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