# strategies/location_strategy.py

import torch
from typing import Optional, Tuple, Union
from core.interfaces import ILocationEstimator
from indoor_location_stage._common import grid_tools
from indoor_location_stage.proposed import soft_em_utils

from indoor_location_stage.baseline.estimator import BaselineEstimator
from indoor_location_stage.proposed.estimator import ProposedEstimator

class BaselineLocationStrategy(ILocationEstimator):
    """
    Strategy for Baseline localization (Hard EM / Viterbi).
    Directly wraps the legacy function.
    """
    def __init__(self, config, device, reference_grid, directions_vectors):
        self.config = config
        self.device = device
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors

        # Pre-calculate AP info ONCE during initialization
        self.num_ap = len(config['ACCESS_POINTS'])
        self.ap_data = {
            'locations': grid_tools.get_ap_locations(config, self.num_ap, device),
            'orientations': torch.tensor(
                [data.get('ORIENTATION_DEG', 0) for data in config['ACCESS_POINTS'].values()], 
                dtype=torch.float32, 
                device=device
            )
        }

    def estimate(
        self, 
        features: torch.Tensor, 
        raw_csi_block: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        estimator = BaselineEstimator(
            features=features,
            config=self.config,
            reference_grid=self.reference_grid,
            ap_data=self.ap_data, 
            device=self.device
        )
        
        trajectory = estimator.solve()
            
        return trajectory

class ProposedLocationStrategy(ILocationEstimator):
    """
    Strategy for Proposed localization (Physics-Aware + optional MLP gating)
    """
    def __init__(self, config, device, reference_grid, directions_vectors):
        self.config = config
        self.device = device
        self.reference_grid = reference_grid
        self.directions_vectors = directions_vectors

        self.gating_model: Optional[torch.nn.Module] = None
        self._pending_gating_state_dict: Optional[dict] = None

        # Pre-calculate AP info ONCE during initialization
        self.num_ap = len(config['ACCESS_POINTS'])

        ap_locations = grid_tools.get_ap_locations(config, self.num_ap, device)
        ap_orientations = torch.tensor(
            [data.get('ORIENTATION_DEG', 0) for data in config['ACCESS_POINTS'].values()], 
            dtype=torch.float32, 
            device=device
        )

        grid_angle_qg = soft_em_utils.calculate_grid_angle_qg(
            reference_grid, ap_locations, ap_orientations
        ).to(device)

        grid_delay_qg = soft_em_utils.calculate_grid_delay_qg(
            reference_grid, ap_locations
        ).to(device)

        G_index = torch.arange(reference_grid.shape[0], device=device)
        neighbor_matrix = grid_tools.get_all_neighbor_indices(config, G_index, device)

        grid_neighbor_delay_diff_qgk = soft_em_utils.calculate_grid_neighbor_delay_diff_qgk(
            grid_delay_qg=grid_delay_qg,
            neighbor_index_matrix=neighbor_matrix,
        ).to(device)

        self.ap_data = {
            'locations': ap_locations,
            'orientations': ap_orientations,
            'grid_angle_qg': grid_angle_qg,
            'grid_delay_qg': grid_delay_qg,
            'neighbor_matrix': neighbor_matrix, 
            'grid_neighbor_delay_diff_qgk': grid_neighbor_delay_diff_qgk,
        }
    
    # =========================================================
    # Public API for INFER_Worker
    # =========================================================
    def set_gating_state_dict(self, state_dict: Optional[dict]) -> None:
        """
        INFER_Worker will call this when it receives a new model update from queues['model'].
        - state_dict is stored and will be applied lazily.
        - if state_dict is None: disable gating.
        """
        self._pending_gating_state_dict = state_dict
        if state_dict is None:
            self.gating_model = None

    # =========================================================
    # Main estimation
    # =========================================================
    def estimate(
        self, 
        features: torch.Tensor, 
        raw_csi_block: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes the Physics-Aware pipeline
        """
        self._ensure_gating_model_ready()

        # Get gating weight
        emission_gating: Optional[torch.Tensor] = None
        transition_gating: Optional[torch.Tensor] = None
        if self.gating_model is not None and raw_csi_block is not None:
            emission_gating, transition_gating = self._inference_gating_model(raw_csi_block)
        
        # Instantiate the Proposed Estimator
        estimator = ProposedEstimator(
            features=features,
            config=self.config,
            reference_grid=self.reference_grid,
            ap_data=self.ap_data,
            device=self.device,
            emission_gating=emission_gating, 
            transition_gating=transition_gating
        )

        trajectory = estimator.solve()
            
        return trajectory

    # =========================================================
    # Internal: lazy build/load gating model
    # =========================================================
    def _ensure_gating_model_ready(self) -> None:
        """
        Lazy init/load:
        - If no pending state_dict -> do nothing (pure physics path)
        - If pending state_dict exists -> build model (if needed) + load + eval
        """
        if self._pending_gating_state_dict is None:
            return

        state_dict = self._pending_gating_state_dict
        self._pending_gating_state_dict = None  # consume once

        if state_dict is None:
            self.gating_model = None
            return

        if self.gating_model is None:
            self.gating_model = self._build_gating_model().to(self.device)

        self.gating_model.load_state_dict(state_dict)
        self.gating_model.eval()

    def _build_gating_model(self) -> torch.nn.Module:
        """
        IMPORTANT:
        You haven't provided the MLP model definition yet.
        Keep this as a hook. When you implement mlp_worker, define the same model here.
        """
        raise NotImplementedError(
            "[ProposedLocationStrategy] gating model is not implemented yet. "
            "Pure physics path is OK. If you start sending state_dict, "
            "please implement _build_gating_model() with the correct architecture."
        )

    # =========================================================
    # Inference
    # =========================================================
    @torch.no_grad()
    def _inference_gating_model(self,raw_csi_block: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.gating_model is None:
            return None, None

        model_device = next(self.gating_model.parameters()).device        
        x = raw_csi_block.to(model_device)

        out = self.gating_model(x)
        
        emission_gating, transition_gating = self._parse_gating_outputs(
            out=out,
            Q=raw_csi_block.size(0),
            T=raw_csi_block.size(1),
            device=model_device,
            dtype=torch.float32,
        )

        if emission_gating is not None:
            emission_gating = emission_gating.clamp(0.0, 1.0).to(self.device)

        if transition_gating is not None:
            transition_gating = transition_gating.clamp(0.0, 1.0).to(self.device)

        return emission_gating, transition_gating


    def _parse_gating_outputs(
        self,
        out: Union[
            None,
            torch.Tensor,
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        ],
        Q: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Normalize gating model outputs into:
        emission_gating: Optional[Tensor] with shape (Q, T) or None
        transition_gating: Optional[Tensor] with shape (Q, T-1) or None

        Supports:
        - None -> (None, None)
        - tuple (eg, tg) where each can be Tensor or None
        - single tensor:
            (Q,T,2) -> eg=x[...,0], tg=x[...,1] (tg trimmed to (Q,T-1))
            (Q,T,1) -> eg only, tg=None
            (Q,T)   -> eg only, tg=None

        HARD GUARANTEE:
        - emission_gating is either None or shape == (Q, T)
        - transition_gating is either None or shape == (Q, T-1)
        """

        # -------------------------
        # Case: out is None
        # -------------------------
        if out is None:
            return None, None

        emission_gating: Optional[torch.Tensor] = None
        transition_gating: Optional[torch.Tensor] = None

        # -------------------------
        # Case: tuple output (ALLOW None in either slot)
        # -------------------------
        if isinstance(out, tuple):
            if len(out) != 2:
                raise ValueError(f"gating_model tuple output must have 2 elements, got {len(out)}")

            eg, tg = out

            if eg is not None and not isinstance(eg, torch.Tensor):
                raise TypeError("emission_gating in tuple must be a torch.Tensor or None")
            if tg is not None and not isinstance(tg, torch.Tensor):
                raise TypeError("transition_gating in tuple must be a torch.Tensor or None")

            emission_gating, transition_gating = eg, tg

        # -------------------------
        # Case: tensor output
        # -------------------------
        else:
            if not isinstance(out, torch.Tensor):
                raise TypeError("gating_model output must be a torch.Tensor, a tuple, or None")

            x = out.to(device=device, dtype=dtype)

            if x.dim() == 2:
                # (Q,T) => emission only
                emission_gating = x
                transition_gating = None

            elif x.dim() == 3 and x.size(-1) == 1:
                # (Q,T,1) => emission only
                emission_gating = x.squeeze(-1)
                transition_gating = None

            elif x.dim() == 3 and x.size(-1) == 2:
                # (Q,T,2) => both (transition given as (Q,T) then trim)
                emission_gating = x[..., 0]
                transition_gating = x[..., 1]
            else:
                raise ValueError(f"Unsupported gating tensor output shape: {tuple(x.shape)}")

        # -------------------------
        # Normalize emission_gating -> (Q,T) or None
        # -------------------------
        if emission_gating is not None:
            emission_gating = emission_gating.to(device=device, dtype=dtype)

            if emission_gating.dim() == 3 and emission_gating.size(-1) == 1:
                emission_gating = emission_gating.squeeze(-1)

            if emission_gating.shape != (Q, T):
                raise ValueError(
                    f"emission_gating shape mismatch: expected (Q,T)=({Q},{T}), got {tuple(emission_gating.shape)}"
                )

        # -------------------------
        # Normalize transition_gating -> (Q,T-1) or None
        # -------------------------
        if transition_gating is not None:
            transition_gating = transition_gating.to(device=device, dtype=dtype)

            if transition_gating.dim() == 3 and transition_gating.size(-1) == 1:
                transition_gating = transition_gating.squeeze(-1)

            # If model returns (Q,T), trim to (Q,T-1)
            if transition_gating.shape == (Q, T):
                transition_gating = transition_gating[:, :-1]

            if transition_gating.shape != (Q, T - 1):
                raise ValueError(
                    f"transition_gating shape mismatch: expected (Q,T-1)=({Q},{T-1}), got {tuple(transition_gating.shape)}"
                )

        return emission_gating, transition_gating


