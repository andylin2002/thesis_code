# engines/neural_engine/stages/represent.py

import torch
from typing import Any, Dict, List, Tuple


class RepresentStage:
    """Build pattern from aggregated CSI."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.num_aps = len(config["ACCESS_POINTS"])
        self.num_rx_antennas = int(config["CSI_DIMENSIONS"]["NUM_RX_ANTENNAS"])
        self.num_subcarriers = int(config["CSI_DIMENSIONS"]["NUM_SUBCARRIERS"])

        if self.num_rx_antennas < 2:
            raise ValueError("NUM_RX_ANTENNAS must be at least 2")

        self.antenna_pairs = self._build_antenna_pairs(self.num_rx_antennas)
        self.num_phase_diff_channels = len(self.antenna_pairs)
        self.num_pattern_channels = 1 + self.num_phase_diff_channels

        self.center_phase_over_subcarriers = bool(
            config.get("NEURAL_CENTER_PHASE_OVER_SUBCARRIERS", True)
        )

    def process(self, aggregated_csi: torch.Tensor) -> torch.Tensor:
        if aggregated_csi.device != self.device:
            aggregated_csi = aggregated_csi.to(self.device, non_blocking=True)

        if aggregated_csi.ndim != 5:
            raise ValueError(
                f"Expected aggregated_csi with shape [B, Q, T, N, M], got {tuple(aggregated_csi.shape)}"
            )

        _, num_aps, _, num_antennas, num_subcarriers = aggregated_csi.shape

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

        return self._build_pattern(aggregated_csi)

    def _build_antenna_pairs(self, num_antennas: int) -> List[Tuple[int, int]]:
        pairs = []
        for i in range(num_antennas):
            for j in range(i + 1, num_antennas):
                pairs.append((i, j))
        return pairs

    def _build_pattern(self, aggregated_csi: torch.Tensor) -> torch.Tensor:
        eps = 1e-8

        # amplitude
        amplitude = torch.abs(aggregated_csi)  # [B, Q, T, N, M]

        amp_mean = amplitude.mean(dim=(-1, -2), keepdim=True).clamp_min(eps)
        amplitude_norm = amplitude / amp_mean

        amplitude_channel = amplitude_norm.mean(dim=3)      # [B, Q, T, M]
        amplitude_channel = amplitude_channel.unsqueeze(3)  # [B, Q, T, 1, M]

        amp_sub_mean = amplitude_channel.mean(dim=-1, keepdim=True)
        amp_sub_std = amplitude_channel.std(dim=-1, keepdim=True).clamp_min(eps)
        amplitude_channel = (amplitude_channel - amp_sub_mean) / amp_sub_std

        # phase difference
        phase = torch.angle(aggregated_csi)  # [B, Q, T, N, M]
        phase_diff_channels = []

        for i, j in self.antenna_pairs:
            phase_diff = phase[:, :, :, j, :] - phase[:, :, :, i, :]  # [B, Q, T, M]

            wrapped = torch.atan2(
                torch.sin(phase_diff),
                torch.cos(phase_diff),
            )

            wrapped = wrapped / torch.pi  # [-1, 1]

            if self.center_phase_over_subcarriers:
                wrapped = wrapped - wrapped.mean(dim=-1, keepdim=True)

            phase_diff_channels.append(wrapped.unsqueeze(3))  # [B, Q, T, 1, M]

        phase_diff_tensor = torch.cat(phase_diff_channels, dim=3)  # [B, Q, T, Cp, M]

        pattern = torch.cat(
            [amplitude_channel, phase_diff_tensor],
            dim=3,
        )  # [B, Q, T, C, M]

        return pattern