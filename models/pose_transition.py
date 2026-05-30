"""Camera-action latent transition probe for pose-chain groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioning import SatelliteMemoryState
from models.encoders.perspective_position_encoder import compute_sat_patch_perspective_uv
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder


CAMERA_ID_FRONT = 0
CAMERA_ID_LEFT_FISHEYE = 1
CAMERA_ID_RIGHT_FISHEYE = 2
CAMERA_ID_UNKNOWN = 3

VIEW_TYPE_FRONT = 0
VIEW_TYPE_VIRTUAL = 1

CAMERA_ACTION_CONTINUOUS_DIM = 26


@dataclass(frozen=True)
class TransitionPairBatch:
    """Flattened transition pairs from a pose-chain batch."""

    group_index: torch.Tensor
    source_view: torch.Tensor
    target_view: torch.Tensor
    flat_source: torch.Tensor
    flat_target: torch.Tensor
    pair_type: Tuple[str, ...]

    @property
    def num_pairs(self) -> int:
        return int(self.flat_source.numel())


@dataclass(frozen=True)
class CameraActionBatch:
    """Continuous and categorical camera action features for A -> B."""

    continuous: torch.Tensor
    source_camera_id: torch.Tensor
    target_camera_id: torch.Tensor
    source_view_type: torch.Tensor
    target_view_type: torch.Tensor


def rotation_matrix_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Return the first two rotation columns as a stable 6D representation."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end with [3,3], got {tuple(rotation.shape)}")
    return rotation[..., :, :2].reshape(*rotation.shape[:-2], 6)


def build_adjacent_transition_pairs(
    batch_size: int,
    num_views: int,
    *,
    device: torch.device,
) -> TransitionPairBatch:
    if int(num_views) < 2:
        raise ValueError("pose-chain transition probe requires at least two views")
    group_index = torch.arange(int(batch_size), device=device).repeat_interleave(int(num_views) - 1)
    source_view = torch.arange(int(num_views) - 1, device=device).repeat(int(batch_size))
    target_view = source_view + 1
    flat_source = group_index * int(num_views) + source_view
    flat_target = group_index * int(num_views) + target_view
    pair_type = tuple("front_to_side" if int(src.item()) == 0 else "side_to_side" for src in source_view)
    return TransitionPairBatch(
        group_index=group_index,
        source_view=source_view,
        target_view=target_view,
        flat_source=flat_source,
        flat_target=flat_target,
        pair_type=pair_type,
    )


def build_transition_pairs_from_indices(
    *,
    batch_size: int,
    num_views: int,
    source_view: torch.Tensor,
    target_view: torch.Tensor,
    device: torch.device,
) -> TransitionPairBatch:
    if source_view.shape != target_view.shape:
        raise ValueError("source_view and target_view must have the same shape")
    source_view = source_view.to(device=device, dtype=torch.long).reshape(-1)
    target_view = target_view.to(device=device, dtype=torch.long).reshape(-1)
    if bool(((source_view < 0) | (source_view >= int(num_views))).any().item()):
        raise ValueError("source_view contains an index outside the chain")
    if bool(((target_view < 0) | (target_view >= int(num_views))).any().item()):
        raise ValueError("target_view contains an index outside the chain")
    group_index = torch.arange(int(batch_size), device=device).repeat_interleave(source_view.numel())
    src = source_view.repeat(int(batch_size))
    dst = target_view.repeat(int(batch_size))
    flat_source = group_index * int(num_views) + src
    flat_target = group_index * int(num_views) + dst
    pair_type = tuple("front_to_side" if int(s.item()) == 0 else "side_to_side" for s in src)
    return TransitionPairBatch(
        group_index=group_index,
        source_view=src,
        target_view=dst,
        flat_source=flat_source,
        flat_target=flat_target,
        pair_type=pair_type,
    )


def _view_name_camera_id(view_name: str) -> int:
    token = str(view_name).lower()
    if token == "front":
        return CAMERA_ID_FRONT
    if token.startswith("yaw_m") or token in {"left", "image_02"}:
        return CAMERA_ID_LEFT_FISHEYE
    if token.startswith("yaw_p") or token in {"right", "image_03"}:
        return CAMERA_ID_RIGHT_FISHEYE
    return CAMERA_ID_UNKNOWN


def _view_name_type(view_name: str) -> int:
    return VIEW_TYPE_FRONT if str(view_name).lower() == "front" else VIEW_TYPE_VIRTUAL


def _view_names_from_batch(
    view_names: Optional[Any],
    *,
    batch_size: int,
    num_views: int,
) -> List[List[str]]:
    if view_names is None:
        return [["front" if view == 0 else f"yaw_unknown_{view}" for view in range(num_views)] for _ in range(batch_size)]
    if isinstance(view_names, Sequence) and len(view_names) == batch_size:
        normalized: List[List[str]] = []
        for group in view_names:
            if isinstance(group, Sequence) and not isinstance(group, (str, bytes)):
                names = [str(item) for item in group]
            else:
                names = [str(group)]
            if len(names) != num_views:
                raise ValueError(f"view_names group must contain {num_views} entries, got {names}")
            normalized.append(names)
        return normalized
    raise ValueError(f"view_names must be a list of {batch_size} view-name lists")


def view_name_ids(
    view_names: Optional[Any],
    *,
    batch_size: int,
    num_views: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    names = _view_names_from_batch(view_names, batch_size=batch_size, num_views=num_views)
    camera_ids = torch.tensor(
        [[_view_name_camera_id(name) for name in group] for group in names],
        dtype=torch.long,
        device=device,
    )
    view_types = torch.tensor(
        [[_view_name_type(name) for name in group] for group in names],
        dtype=torch.long,
        device=device,
    )
    return camera_ids, view_types


def _flatten_group_tensor(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    trailing_dims: Tuple[int, ...],
    name: str,
) -> torch.Tensor:
    expected_shape = (int(batch_size), int(num_views), *trailing_dims)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor.reshape(int(batch_size) * int(num_views), *trailing_dims)


def _normalize_intrinsics(K: torch.Tensor, *, image_hw: Tuple[int, int]) -> torch.Tensor:
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    width = float(max(1, image_w))
    height = float(max(1, image_h))
    return torch.stack(
        [
            K[:, 0, 0] / width,
            K[:, 1, 1] / height,
            K[:, 0, 2] / width,
            K[:, 1, 2] / height,
        ],
        dim=-1,
    )


def _yaw_features(yaw_a: torch.Tensor, yaw_b: torch.Tensor) -> torch.Tensor:
    yaw_a = torch.nan_to_num(yaw_a.float(), nan=0.0)
    yaw_b = torch.nan_to_num(yaw_b.float(), nan=0.0)
    yaw_a_rad = yaw_a * torch.pi / 180.0
    yaw_b_rad = yaw_b * torch.pi / 180.0
    delta_rad = yaw_b_rad - yaw_a_rad
    return torch.stack(
        [
            torch.sin(yaw_a_rad),
            torch.cos(yaw_a_rad),
            torch.sin(yaw_b_rad),
            torch.cos(yaw_b_rad),
            torch.sin(delta_rad),
            torch.cos(delta_rad),
            delta_rad / torch.pi,
        ],
        dim=-1,
    )


def build_camera_action_batch(
    *,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    camera_height_m: torch.Tensor,
    vehicle_yaw_degs: Optional[torch.Tensor],
    view_names: Optional[Any],
    pairs: TransitionPairBatch,
    image_hw: Tuple[int, int],
) -> CameraActionBatch:
    """Build full camera-action features for each A -> B pair."""
    if K.ndim != 4 or T_cam_to_world.ndim != 4:
        raise ValueError("K must be [B,V,3,3] and T_cam_to_world must be [B,V,4,4]")
    batch_size, num_views = int(K.shape[0]), int(K.shape[1])
    device = K.device
    flat_K = _flatten_group_tensor(K, batch_size=batch_size, num_views=num_views, trailing_dims=(3, 3), name="K")
    flat_T = _flatten_group_tensor(
        T_cam_to_world,
        batch_size=batch_size,
        num_views=num_views,
        trailing_dims=(4, 4),
        name="T_cam_to_world",
    )
    flat_height = camera_height_m.reshape(batch_size * num_views).to(device=device, dtype=K.dtype)
    if vehicle_yaw_degs is None:
        flat_yaw = torch.zeros(batch_size * num_views, device=device, dtype=K.dtype)
    else:
        flat_yaw = vehicle_yaw_degs.reshape(batch_size * num_views).to(device=device, dtype=K.dtype)
    camera_ids, view_types = view_name_ids(
        view_names,
        batch_size=batch_size,
        num_views=num_views,
        device=device,
    )
    flat_camera_ids = camera_ids.reshape(batch_size * num_views)
    flat_view_types = view_types.reshape(batch_size * num_views)

    source = pairs.flat_source.to(device=device)
    target = pairs.flat_target.to(device=device)
    T_a = flat_T.index_select(0, source).float()
    T_b = flat_T.index_select(0, target).float()
    T_ab = torch.linalg.inv(T_a) @ T_b
    se3 = torch.cat([rotation_matrix_to_6d(T_ab[:, :3, :3]), T_ab[:, :3, 3]], dim=-1)
    K_a = _normalize_intrinsics(flat_K.index_select(0, source).float(), image_hw=image_hw)
    K_b = _normalize_intrinsics(flat_K.index_select(0, target).float(), image_hw=image_hw)
    heights = torch.stack(
        [
            flat_height.index_select(0, source).float(),
            flat_height.index_select(0, target).float(),
        ],
        dim=-1,
    )
    yaw = _yaw_features(flat_yaw.index_select(0, source), flat_yaw.index_select(0, target))
    continuous = torch.cat([se3, K_a, K_b, heights, yaw], dim=-1)
    if continuous.shape[-1] != CAMERA_ACTION_CONTINUOUS_DIM:
        raise RuntimeError(
            f"camera action continuous dim must be {CAMERA_ACTION_CONTINUOUS_DIM}, got {continuous.shape[-1]}"
        )
    return CameraActionBatch(
        continuous=continuous,
        source_camera_id=flat_camera_ids.index_select(0, source),
        target_camera_id=flat_camera_ids.index_select(0, target),
        source_view_type=flat_view_types.index_select(0, source),
        target_view_type=flat_view_types.index_select(0, target),
    )


def project_target_satellite_geometry(
    *,
    sat_state: SatelliteMemoryState,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    camera_height_m: torch.Tensor,
    pairs: TransitionPairBatch,
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project shared satellite patch coordinates into each pair's target view."""
    if sat_state.bev_coords is None:
        raise ValueError("satellite state must include bev_coords for target geometry")
    batch_size, num_views = int(K.shape[0]), int(K.shape[1])
    device = K.device
    flat_K = _flatten_group_tensor(K, batch_size=batch_size, num_views=num_views, trailing_dims=(3, 3), name="K")
    flat_T_cam = _flatten_group_tensor(
        T_cam_to_world,
        batch_size=batch_size,
        num_views=num_views,
        trailing_dims=(4, 4),
        name="T_cam_to_world",
    )
    flat_T_imu = _flatten_group_tensor(
        T_imu_to_world,
        batch_size=batch_size,
        num_views=num_views,
        trailing_dims=(4, 4),
        name="T_imu_to_world",
    )
    flat_height = camera_height_m.reshape(batch_size * num_views).to(device=device)
    bev_coords = sat_state.bev_coords.index_select(0, pairs.group_index.to(device=sat_state.bev_coords.device))
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    target = pairs.flat_target.to(device=device)
    return compute_sat_patch_perspective_uv(
        bev_coords=bev_coords,
        K=flat_K.index_select(0, target),
        T_cam_to_world=flat_T_cam.index_select(0, target),
        T_imu_to_world=flat_T_imu.index_select(0, target),
        camera_height_m=flat_height.index_select(0, target),
        image_w=image_w,
        image_h=image_h,
    )


def _pool_target_geometry(uv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_f = valid.to(dtype=uv.dtype).unsqueeze(-1)
    count = valid_f.sum(dim=1).clamp_min(1.0)
    mean = (uv * valid_f).sum(dim=1) / count
    centered = (uv - mean.unsqueeze(1)) * valid_f
    std = (centered.pow(2).sum(dim=1) / count).sqrt()
    valid_ratio = valid.to(dtype=uv.dtype).float().mean(dim=1, keepdim=True)
    return torch.cat([mean, std, valid_ratio.to(dtype=uv.dtype)], dim=-1)


def _transition_loss(pred: torch.Tensor, target: torch.Tensor, mse_weight: float) -> torch.Tensor:
    return F.l1_loss(pred.float(), target.float()) + float(mse_weight) * F.mse_loss(pred.float(), target.float())


def _resolve_transition_latents(
    *,
    latents: Optional[torch.Tensor],
    source_latents: Optional[torch.Tensor],
    target_latents: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if source_latents is None:
        if latents is None:
            raise ValueError("transition auxiliary requires latents or source_latents")
        source_latents = latents
    if target_latents is None:
        target_latents = latents if latents is not None else source_latents
    if source_latents.ndim != 4:
        raise ValueError(f"source_latents must be flattened [B*V,C,H,W], got {tuple(source_latents.shape)}")
    if target_latents.ndim != 4:
        raise ValueError(f"target_latents must be flattened [B*V,C,H,W], got {tuple(target_latents.shape)}")
    if tuple(source_latents.shape) != tuple(target_latents.shape):
        raise ValueError(
            "source_latents and target_latents must have the same flattened shape, "
            f"got {tuple(source_latents.shape)} and {tuple(target_latents.shape)}"
        )
    return source_latents, target_latents


class CameraActionEncoder(nn.Module):
    """Embed relative SE(3), intrinsics, camera identity, and yaw features."""

    def __init__(
        self,
        *,
        output_dim: int = 128,
        camera_embed_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.camera_embedding = nn.Embedding(4, camera_embed_dim)
        self.view_type_embedding = nn.Embedding(2, camera_embed_dim)
        input_dim = CAMERA_ACTION_CONTINUOUS_DIM + camera_embed_dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = int(output_dim)

    def forward(self, action: CameraActionBatch) -> torch.Tensor:
        categorical = torch.cat(
            [
                self.camera_embedding(action.source_camera_id),
                self.camera_embedding(action.target_camera_id),
                self.view_type_embedding(action.source_view_type),
                self.view_type_embedding(action.target_view_type),
            ],
            dim=-1,
        )
        return self.net(torch.cat([action.continuous.float(), categorical], dim=-1))


class TransitionHead(nn.Module):
    """Predict target VAE latent from source latent and camera-action context."""

    def __init__(
        self,
        *,
        latent_channels: int = 4,
        sat_token_dim: int = 1024,
        action_dim: int = 128,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.action_encoder = CameraActionEncoder(output_dim=action_dim)
        self.sat_proj = nn.Sequential(
            nn.LayerNorm(sat_token_dim),
            nn.Linear(sat_token_dim, action_dim),
            nn.SiLU(),
        )
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, action_dim),
            nn.SiLU(),
        )
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(action_dim * 3),
            nn.Linear(action_dim * 3, hidden_channels),
            nn.SiLU(),
        )
        self.in_conv = nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.GroupNorm(8 if hidden_channels % 8 == 0 else 1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8 if hidden_channels % 8 == 0 else 1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(
        self,
        source_latents: torch.Tensor,
        action: CameraActionBatch,
        *,
        sat_tokens: torch.Tensor,
        target_geometry_uv: torch.Tensor,
        target_geometry_valid: torch.Tensor,
    ) -> torch.Tensor:
        action_emb = self.action_encoder(action)
        sat_emb = self.sat_proj(sat_tokens.float().mean(dim=1))
        geom_emb = self.geometry_proj(_pool_target_geometry(target_geometry_uv.float(), target_geometry_valid.bool()))
        cond = self.cond_proj(torch.cat([action_emb, sat_emb, geom_emb], dim=-1))
        h = self.in_conv(source_latents.float())
        h = h + cond[:, :, None, None]
        h = h + self.block(h)
        return source_latents.float() + self.out_conv(h)


def predict_transition_pairs(
    transition_head: TransitionHead,
    *,
    latents: torch.Tensor,
    sat_state: SatelliteMemoryState,
    batch: Mapping[str, Any],
    pairs: TransitionPairBatch,
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, CameraActionBatch, torch.Tensor, torch.Tensor]:
    K = batch["K"]
    T_cam_to_world = batch["T_cam_to_world"]
    T_imu_to_world = batch["T_imu_to_world"]
    camera_height_m = batch["camera_height_m"]
    action = build_camera_action_batch(
        K=K,
        T_cam_to_world=T_cam_to_world,
        camera_height_m=camera_height_m,
        vehicle_yaw_degs=batch.get("vehicle_yaw_degs"),
        view_names=batch.get("view_names"),
        pairs=pairs,
        image_hw=image_hw,
    )
    target_uv, target_valid = project_target_satellite_geometry(
        sat_state=sat_state,
        K=K,
        T_cam_to_world=T_cam_to_world,
        T_imu_to_world=T_imu_to_world,
        camera_height_m=camera_height_m,
        pairs=pairs,
        image_hw=image_hw,
    )
    sat_tokens = sat_state.tokens.index_select(0, pairs.group_index.to(device=sat_state.tokens.device))
    source_latents = latents.index_select(0, pairs.flat_source.to(device=latents.device))
    pred = transition_head(
        source_latents,
        action,
        sat_tokens=sat_tokens,
        target_geometry_uv=target_uv,
        target_geometry_valid=target_valid,
    )
    return pred, action, target_uv, target_valid


def compute_transition_auxiliary_outputs(
    transition_head: TransitionHead,
    *,
    latents: Optional[torch.Tensor] = None,
    source_latents: Optional[torch.Tensor] = None,
    target_latents: Optional[torch.Tensor] = None,
    sat_state: SatelliteMemoryState,
    batch: Mapping[str, Any],
    image_hw: Tuple[int, int],
    cycle_weight: float = 0.1,
    composition_weight: float = 0.05,
    mse_weight: float = 0.1,
) -> Mapping[str, torch.Tensor]:
    source_latents, target_latents = _resolve_transition_latents(
        latents=latents,
        source_latents=source_latents,
        target_latents=target_latents,
    )
    if batch["K"].ndim != 4:
        raise ValueError("pose transition auxiliary requires grouped K with shape [B,V,3,3]")
    batch_size, num_views = int(batch["K"].shape[0]), int(batch["K"].shape[1])
    pairs = build_adjacent_transition_pairs(batch_size, num_views, device=source_latents.device)
    pred_b, _, _, _ = predict_transition_pairs(
        transition_head,
        latents=source_latents,
        sat_state=sat_state,
        batch=batch,
        pairs=pairs,
        image_hw=image_hw,
    )
    target_b = target_latents.index_select(0, pairs.flat_target.to(device=target_latents.device)).to(device=pred_b.device)
    per_pair_loss = (pred_b.float() - target_b.float()).abs().mean(dim=(1, 2, 3))
    transition_loss = _transition_loss(pred_b, target_b, mse_weight)

    front_mask = torch.tensor(
        [pair_type == "front_to_side" for pair_type in pairs.pair_type],
        dtype=torch.bool,
        device=source_latents.device,
    )
    side_mask = ~front_mask
    front_to_side_loss = per_pair_loss[front_mask].mean() if bool(front_mask.any().item()) else transition_loss * 0.0
    side_to_side_loss = per_pair_loss[side_mask].mean() if bool(side_mask.any().item()) else transition_loss * 0.0

    reverse_pairs = TransitionPairBatch(
        group_index=pairs.group_index,
        source_view=pairs.target_view,
        target_view=pairs.source_view,
        flat_source=pairs.flat_target,
        flat_target=pairs.flat_source,
        pair_type=pairs.pair_type,
    )
    reverse_uv, reverse_valid = project_target_satellite_geometry(
        sat_state=sat_state,
        K=batch["K"],
        T_cam_to_world=batch["T_cam_to_world"],
        T_imu_to_world=batch["T_imu_to_world"],
        camera_height_m=batch["camera_height_m"],
        pairs=reverse_pairs,
        image_hw=image_hw,
    )
    pred_back = transition_head(
        pred_b,
        build_camera_action_batch(
            K=batch["K"],
            T_cam_to_world=batch["T_cam_to_world"],
            camera_height_m=batch["camera_height_m"],
            vehicle_yaw_degs=batch.get("vehicle_yaw_degs"),
            view_names=batch.get("view_names"),
            pairs=reverse_pairs,
            image_hw=image_hw,
        ),
        sat_tokens=sat_state.tokens.index_select(0, reverse_pairs.group_index.to(device=sat_state.tokens.device)),
        target_geometry_uv=reverse_uv,
        target_geometry_valid=reverse_valid,
    )
    source_a = target_latents.index_select(0, pairs.flat_source.to(device=target_latents.device)).to(device=pred_back.device)
    cycle_loss = F.l1_loss(pred_back.float(), source_a.float())

    composition_loss = transition_loss * 0.0
    if num_views >= 3:
        source = torch.arange(num_views - 2, device=source_latents.device, dtype=torch.long)
        mid = source + 1
        target = source + 2
        pairs_ab = build_transition_pairs_from_indices(
            batch_size=batch_size,
            num_views=num_views,
            source_view=source,
            target_view=mid,
            device=source_latents.device,
        )
        pairs_bc = build_transition_pairs_from_indices(
            batch_size=batch_size,
            num_views=num_views,
            source_view=mid,
            target_view=target,
            device=source_latents.device,
        )
        pairs_ac = build_transition_pairs_from_indices(
            batch_size=batch_size,
            num_views=num_views,
            source_view=source,
            target_view=target,
            device=source_latents.device,
        )
        pred_mid, _, _, _ = predict_transition_pairs(
            transition_head,
            latents=source_latents,
            sat_state=sat_state,
            batch=batch,
            pairs=pairs_ab,
            image_hw=image_hw,
        )
        action_bc = build_camera_action_batch(
            K=batch["K"],
            T_cam_to_world=batch["T_cam_to_world"],
            camera_height_m=batch["camera_height_m"],
            vehicle_yaw_degs=batch.get("vehicle_yaw_degs"),
            view_names=batch.get("view_names"),
            pairs=pairs_bc,
            image_hw=image_hw,
        )
        uv_bc, valid_bc = project_target_satellite_geometry(
            sat_state=sat_state,
            K=batch["K"],
            T_cam_to_world=batch["T_cam_to_world"],
            T_imu_to_world=batch["T_imu_to_world"],
            camera_height_m=batch["camera_height_m"],
            pairs=pairs_bc,
            image_hw=image_hw,
        )
        pred_via_mid = transition_head(
            pred_mid,
            action_bc,
            sat_tokens=sat_state.tokens.index_select(0, pairs_bc.group_index.to(device=sat_state.tokens.device)),
            target_geometry_uv=uv_bc,
            target_geometry_valid=valid_bc,
        )
        pred_direct, _, _, _ = predict_transition_pairs(
            transition_head,
            latents=source_latents,
            sat_state=sat_state,
            batch=batch,
            pairs=pairs_ac,
            image_hw=image_hw,
        )
        composition_loss = F.l1_loss(pred_via_mid.float(), pred_direct.float())

    total_loss = transition_loss + float(cycle_weight) * cycle_loss + float(composition_weight) * composition_loss
    return {
        "loss": total_loss,
        "transition_loss": transition_loss.detach(),
        "front_to_side_loss": front_to_side_loss.detach(),
        "side_to_side_loss": side_to_side_loss.detach(),
        "cycle_loss": cycle_loss.detach(),
        "composition_loss": composition_loss.detach(),
        "num_pairs": torch.tensor(float(pairs.num_pairs), device=source_latents.device),
    }


class PoseTransitionProbe(nn.Module):
    """Latent transition probe using shared satellite memory and full camera action."""

    def __init__(
        self,
        satellite_encoder: SatelliteConditionEncoder,
        transition_head: TransitionHead,
        *,
        cycle_weight: float = 0.1,
        composition_weight: float = 0.05,
        mse_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.satellite_encoder = satellite_encoder
        self.transition_head = transition_head
        self.cycle_weight = float(cycle_weight)
        self.composition_weight = float(composition_weight)
        self.mse_weight = float(mse_weight)

    @staticmethod
    def _transition_loss(pred: torch.Tensor, target: torch.Tensor, mse_weight: float) -> torch.Tensor:
        return _transition_loss(pred, target, mse_weight)

    def encode_satellite(self, sat_images: torch.Tensor) -> SatelliteMemoryState:
        return self.satellite_encoder(sat_images)

    def _predict_pairs(
        self,
        *,
        latents: torch.Tensor,
        sat_state: SatelliteMemoryState,
        batch: Mapping[str, Any],
        pairs: TransitionPairBatch,
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, CameraActionBatch, torch.Tensor, torch.Tensor]:
        return predict_transition_pairs(
            self.transition_head,
            latents=latents,
            sat_state=sat_state,
            batch=batch,
            pairs=pairs,
            image_hw=image_hw,
        )

    def forward(
        self,
        *,
        latents: torch.Tensor,
        sat_images: torch.Tensor,
        batch: Mapping[str, Any],
        image_hw: Tuple[int, int],
    ) -> Mapping[str, torch.Tensor]:
        if latents.ndim != 4:
            raise ValueError(f"latents must be flattened [B*V,C,H,W], got {tuple(latents.shape)}")
        if batch["K"].ndim != 4:
            raise ValueError("pose transition probe requires grouped K with shape [B,V,3,3]")
        sat_state = self.encode_satellite(sat_images)
        return compute_transition_auxiliary_outputs(
            self.transition_head,
            latents=latents,
            sat_state=sat_state,
            batch=batch,
            image_hw=image_hw,
            cycle_weight=self.cycle_weight,
            composition_weight=self.composition_weight,
            mse_weight=self.mse_weight,
        )
