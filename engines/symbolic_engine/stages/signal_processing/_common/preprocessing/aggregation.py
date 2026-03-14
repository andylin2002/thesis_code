# engines/symbolic_engine/stages/signal_processing/_common/preprocessing/aggregation.py

from typing import Dict, Any, Optional
import torch

def run_csi_aggregation(
    raw_csi_data: torch.Tensor,
    config: Dict[str, Any]
) -> Optional[torch.Tensor]:
    """
    Unified preprocessing:
      1) Packeting: (Q, TP, N, M) -> (Q, T, P, N, M)
      2) Aggregation (SVD rank-1): (QT, P, N, M) -> (QT, 1, N, M)
      3) Reshape back: (QT, 1, N, M) -> (Q, T, N, M)

    Args:
        raw_csi_data: Tensor (Q, TP, N, M)
        config: must contain NUM_SAMPLE (T), NUM_PACKET (P), ACCESS_POINTS (for Q)
    Returns:
        processed_csi: Tensor (Q, T, N, M)
    """
    # --- Parameters ---
    ap_data = config.get('ACCESS_POINTS', {})
    num_ap = len(ap_data)
    num_sample = config['NUM_SAMPLE']   # T
    num_packet = config['NUM_PACKET']   # P

    # --- Packeting --- (Q, TP, N, M) -> (Q, T, P, N, M)
    packeted = run_packeting_gpu(
        csi_data=raw_csi_data,
        T_time=num_sample,
        P_packet=num_packet
    )

    # --- Flatten --- (Q, T, P, N, M) -> (QT, P, N, M)
    num_batch = num_sample * num_ap
    non_aggregated = packeted.reshape(
        num_batch,
        num_packet,
        *packeted.shape[3:]
    ).contiguous()

    # --- Aggregation --- (QT, P, N, M) -> (QT, 1, N, M)
    aggregated = run_svd_rank1_aggregation_gpu(non_aggregated)

    # --- Reshape --- (QT, 1, N, M) -> (Q, T, 1, N, M) -> (Q, T, N, M)
    reshaped = aggregated.reshape(
        num_ap,
        num_sample,
        1,
        *aggregated.shape[2:]
    )
    processed_csi = reshaped.squeeze(2).contiguous()  # (Q, T, N, M)

    return processed_csi


def run_packeting_gpu(
    csi_data: torch.Tensor,
    T_time: int,
    P_packet: int
) -> torch.Tensor:
    """
    Reshape time dimension TP into (T, P):
      (Q, TP, N, M) -> (Q, T, P, N, M)
    """
    Q_ap, TP_total, N_antenna, M_subcarrier = csi_data.shape

    # --- Check if the input N_TIME matches T*P ---
    if TP_total != T_time * P_packet:
        raise ValueError(
            f"[PACKETING] Total time dimension ({TP_total}) does not match T*P ({T_time * P_packet})."
        )

    # Reshape the N_TIME dimension into (T, P)
    # The output shape is (Q, T, P, N, M)
    packeted = csi_data.reshape(Q_ap, T_time, P_packet, N_antenna, M_subcarrier)
    return packeted


def run_svd_rank1_aggregation_gpu(
    non_aggregated_csi_gpu: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Rank-1 aggregation via SVD over packet dimension.

    Input:
      non_aggregated_csi_gpu: (B=QT, P, N, M)
    Output:
      aggregated: (B=QT, 1, N, M)

    Notes:
      - This assumes snapshots within the P window are sufficiently consistent
        so a rank-1 approximation is meaningful.
      - If your CSI is complex, torch.linalg.svd supports complex tensors.
    """
    # --- Parameter Setup ---
    B_batch, P_packet, N_antenna, M_subcarrier = non_aggregated_csi_gpu.shape

    # --- Reshape ---
    # (B, P, N*M)
    combined = non_aggregated_csi_gpu.view(B_batch, P_packet, N_antenna * M_subcarrier)
    # (B, N*M, P)
    svd_input = combined.permute(0, 2, 1).contiguous()

    # --- SVD decomposition ---
    U, S, Vh = torch.linalg.svd(svd_input)
    u1 = U[:, :, 0]                # (B, D)
    s1 = S[:, 0].unsqueeze(1)       # (B, 1)
    aggregated_flat = u1 * s1       # (B, D)

    # --- Get Aggregated CSI data ---
    aggregated = aggregated_flat.view(B_batch, 1, N_antenna, M_subcarrier)
    return aggregated
