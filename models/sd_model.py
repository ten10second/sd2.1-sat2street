"""Satellite-conditioned Stable Diffusion model wrappers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler

try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
except ImportError:  # pragma: no cover - compatibility with older diffusers layouts
    from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

from models.conditioning import SatelliteMemoryState
from models.cross_view_refiner import CrossViewLatentRefiner
from models.encoders.perspective_position_encoder import compute_sat_patch_perspective_uv
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.pose_transition import TransitionHead, compute_transition_auxiliary_outputs
from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0, QueryUVSlicedAttnProcessor


logger = logging.getLogger(__name__)


DEFAULT_ATTENTION_ALIGNMENT_LAYERS = (
    "mid_block.attentions.0.transformer_blocks.0.attn2",
)
TRANSITION_AUX_SOURCE_GT_LATENT = "gt_latent"
TRANSITION_AUX_SOURCE_PREDICTED_X0 = "predicted_x0"


def normalize_transition_aux_source(source: str) -> str:
    normalized = str(source).strip().lower().replace("-", "_")
    aliases = {
        "gt": TRANSITION_AUX_SOURCE_GT_LATENT,
        "gt_latent": TRANSITION_AUX_SOURCE_GT_LATENT,
        "gt_latents": TRANSITION_AUX_SOURCE_GT_LATENT,
        "latent": TRANSITION_AUX_SOURCE_GT_LATENT,
        "vae_latent": TRANSITION_AUX_SOURCE_GT_LATENT,
        "x0": TRANSITION_AUX_SOURCE_PREDICTED_X0,
        "pred_x0": TRANSITION_AUX_SOURCE_PREDICTED_X0,
        "predicted_x0": TRANSITION_AUX_SOURCE_PREDICTED_X0,
        "pred_original_sample": TRANSITION_AUX_SOURCE_PREDICTED_X0,
    }
    if normalized not in aliases:
        raise ValueError(
            "transition_aux_source must be one of "
            f"{TRANSITION_AUX_SOURCE_GT_LATENT!r} or {TRANSITION_AUX_SOURCE_PREDICTED_X0!r}, got {source!r}"
        )
    return aliases[normalized]


def _resolve_module_path(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


class SatelliteConditionedUNet(UNet2DConditionModel):
    """Thin UNet wrapper that routes satellite tokens and geometry into cross-attention."""

    _logged_diag: bool = False

    def __init__(
        self,
        query_uv_pe_enabled: bool = False,
        query_geometry_bias_enabled: bool = False,
        query_geometry_bias_scale: float = 2.0,
        query_geometry_invalid_penalty: float = -1e4,
        query_geometry_score_enabled: bool = False,
        query_geometry_score_dim: int = 64,
        query_geometry_score_num_freqs: int = 6,
        query_geometry_score_gate_init: float = 1.0,
        query_geometry_score_layers: Optional[Sequence[str]] = None,
        query_geometry_score_max_query_tokens: Optional[int] = None,
        query_geometry_score_mode: str = "geometry_first_semantic_refine",
        query_geometry_candidate_radius: float = 0.35,
        query_geometry_candidate_min_k: int = 16,
        query_geometry_candidate_invalid_penalty: float = -1e4,
        query_semantic_score_dim: int = 64,
        query_semantic_score_alpha: float = 0.25,
        query_uv_gate_init: float = 0.0,
        attention_alignment_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        del query_uv_pe_enabled, query_geometry_bias_enabled, query_geometry_bias_scale
        del query_geometry_invalid_penalty, query_uv_gate_init
        self.query_uv_pe_enabled = False
        self.query_geometry_bias_enabled = False
        self.query_geometry_score_enabled = bool(query_geometry_score_enabled)
        self.query_geometry_score_dim = int(query_geometry_score_dim)
        self.query_geometry_score_num_freqs = int(query_geometry_score_num_freqs)
        self.query_geometry_score_gate_init = float(query_geometry_score_gate_init)
        self.query_geometry_score_mode = str(query_geometry_score_mode)
        self.query_geometry_candidate_radius = float(query_geometry_candidate_radius)
        self.query_geometry_candidate_min_k = int(query_geometry_candidate_min_k)
        self.query_geometry_candidate_invalid_penalty = float(query_geometry_candidate_invalid_penalty)
        self.query_semantic_score_dim = int(query_semantic_score_dim)
        self.query_semantic_score_alpha = float(query_semantic_score_alpha)
        self.query_geometry_score_layers = (
            None
            if query_geometry_score_layers is None
            else tuple(str(layer) for layer in query_geometry_score_layers)
        )
        self.query_geometry_score_max_query_tokens = (
            None
            if query_geometry_score_max_query_tokens is None
            else int(query_geometry_score_max_query_tokens)
        )
        self.query_geometry_score_runtime_scale = 1.0
        self.query_semantic_score_runtime_alpha = self.query_semantic_score_alpha
        self.attention_alignment_enabled = bool(attention_alignment_enabled)
        self._attention_debug_layers: Optional[Sequence[str]] = None
        self._attention_debug_storage: Optional[Dict[str, Any]] = None
        self._install_query_uv_attention_processors()

    def _build_attention_processors(self):
        if (
            not self.query_geometry_score_enabled
            and not self.attention_alignment_enabled
        ):
            return AttnProcessor2_0()
        return self._build_query_uv_attention_processors()

    def _build_query_uv_attention_processors(self):
        processors = {}
        for name in self.attn_processors.keys():
            layer_name = name.removesuffix(".processor")
            attn_module = _resolve_module_path(self, layer_name)
            query_dim = int(attn_module.to_q.out_features)
            processors[name] = QueryUVAttnProcessor2_0(
                query_dim=query_dim,
                geometry_score_enabled=bool(self.query_geometry_score_enabled and name.endswith(".attn2.processor")),
                geometry_score_dim=self.query_geometry_score_dim,
                geometry_score_num_freqs=self.query_geometry_score_num_freqs,
                geometry_score_gate_init=self.query_geometry_score_gate_init,
                geometry_score_layers=self.query_geometry_score_layers,
                geometry_score_max_query_tokens=self.query_geometry_score_max_query_tokens,
                geometry_score_mode=self.query_geometry_score_mode,
                candidate_radius=self.query_geometry_candidate_radius,
                candidate_min_k=self.query_geometry_candidate_min_k,
                candidate_invalid_penalty=self.query_geometry_candidate_invalid_penalty,
                semantic_score_dim=self.query_semantic_score_dim,
                semantic_score_alpha=self.query_semantic_score_alpha,
                layer_name=layer_name,
            )
        return processors

    def _install_query_uv_attention_processors(self) -> None:
        self.set_attn_processor(self._build_attention_processors())
        self._sync_query_geometry_score_runtime_scale()
        self._sync_query_semantic_score_runtime_alpha()

    def _sync_query_geometry_score_runtime_scale(self) -> None:
        scale = float(getattr(self, "query_geometry_score_runtime_scale", 1.0))
        for processor in self.attn_processors.values():
            setter = getattr(processor, "set_geometry_score_runtime_scale", None)
            if callable(setter):
                setter(scale)

    def _sync_query_semantic_score_runtime_alpha(self) -> None:
        alpha = float(getattr(self, "query_semantic_score_runtime_alpha", self.query_semantic_score_alpha))
        for processor in self.attn_processors.values():
            setter = getattr(processor, "set_semantic_score_runtime_alpha", None)
            if callable(setter):
                setter(alpha)

    def set_query_geometry_score_runtime_scale(self, scale: float) -> None:
        self.query_geometry_score_runtime_scale = float(scale)
        self._sync_query_geometry_score_runtime_scale()

    def set_query_semantic_score_runtime_alpha(self, alpha: float) -> None:
        self.query_semantic_score_runtime_alpha = float(alpha)
        self._sync_query_semantic_score_runtime_alpha()

    def set_attention_slice(self, slice_size="auto"):
        if (
            not self.query_geometry_score_enabled
            and not self.attention_alignment_enabled
        ):
            return super().set_attention_slice(slice_size)

        if slice_size is None:
            self._install_query_uv_attention_processors()
            return

        super().set_attention_slice(slice_size)
        sliced_processors = {}
        for name, processor in self.attn_processors.items():
            slice_value = getattr(processor, "slice_size", None)
            if slice_value is None:
                raise ValueError(
                    f"Unable to preserve query UV attention slicing for {name}: "
                    f"{type(processor).__name__} has no slice_size"
                )
            layer_name = name.removesuffix(".processor")
            attn_module = _resolve_module_path(self, layer_name)
            query_dim = int(attn_module.to_q.out_features)
            sliced_processors[name] = QueryUVSlicedAttnProcessor(
                query_dim=query_dim,
                slice_size=int(slice_value),
                geometry_score_enabled=bool(self.query_geometry_score_enabled and name.endswith(".attn2.processor")),
                geometry_score_dim=self.query_geometry_score_dim,
                geometry_score_num_freqs=self.query_geometry_score_num_freqs,
                geometry_score_gate_init=self.query_geometry_score_gate_init,
                geometry_score_layers=self.query_geometry_score_layers,
                geometry_score_max_query_tokens=self.query_geometry_score_max_query_tokens,
                geometry_score_mode=self.query_geometry_score_mode,
                candidate_radius=self.query_geometry_candidate_radius,
                candidate_min_k=self.query_geometry_candidate_min_k,
                candidate_invalid_penalty=self.query_geometry_candidate_invalid_penalty,
                semantic_score_dim=self.query_semantic_score_dim,
                semantic_score_alpha=self.query_semantic_score_alpha,
                layer_name=layer_name,
            )
        self.set_attn_processor(sliced_processors)
        self._sync_query_geometry_score_runtime_scale()
        self._sync_query_semantic_score_runtime_alpha()

    def enable_attention_debug(
        self,
        *,
        layers: Sequence[str],
        storage: Dict[str, Any],
    ) -> None:
        self._attention_debug_layers = tuple(str(layer) for layer in layers)
        self._attention_debug_storage = storage

    def disable_attention_debug(self) -> None:
        self._attention_debug_layers = None
        self._attention_debug_storage = None

    def is_attention_debug_enabled(self) -> bool:
        return self._attention_debug_layers is not None and self._attention_debug_storage is not None

    @staticmethod
    def _normalize_query_base_hw(cross_attention_kwargs):
        if cross_attention_kwargs is None:
            return None
        if not isinstance(cross_attention_kwargs, dict):
            raise ValueError(
                f"cross_attention_kwargs must be a dict when query-based geometry features are enabled, "
                f"got {type(cross_attention_kwargs)!r}"
            )
        query_base_hw = cross_attention_kwargs.get("query_base_hw")
        if query_base_hw is None:
            return None
        if isinstance(query_base_hw, torch.Tensor):
            if query_base_hw.numel() != 2:
                raise ValueError(f"query_base_hw tensor must contain 2 values, got {tuple(query_base_hw.shape)}")
            query_base_hw = tuple(int(x) for x in query_base_hw.reshape(-1).tolist())
        elif hasattr(query_base_hw, "__len__") and len(query_base_hw) != 2:
            raise ValueError(f"query_base_hw must contain 2 values, got {query_base_hw}")
        return tuple(int(x) for x in query_base_hw)

    def forward(self, *args, sat_tokens: Optional[torch.Tensor] = None, **kwargs):
        encoder_hidden_states = kwargs.pop("encoder_hidden_states", None)
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
        if sat_tokens is not None:
            encoder_hidden_states = sat_tokens
        if encoder_hidden_states is None:
            raise ValueError("SatelliteConditionedUNet requires sat_tokens or encoder_hidden_states")
        attention_alignment_requested = (
            isinstance(cross_attention_kwargs, dict)
            and isinstance(cross_attention_kwargs.get("attention_alignment"), dict)
        )
        if (
            self.query_geometry_score_enabled
            or attention_alignment_requested
            or self.is_attention_debug_enabled()
        ):
            if self._normalize_query_base_hw(cross_attention_kwargs) is None:
                raise ValueError(
                    "query-based geometry features require cross_attention_kwargs['query_base_hw'] "
                    "to thread latent spatial coordinates into cross-attention"
                )
        if self.query_geometry_score_enabled:
            if not isinstance(cross_attention_kwargs, dict) or cross_attention_kwargs.get("sat_perspective_uv") is None:
                raise ValueError(
                    "query geometry features require cross_attention_kwargs['sat_perspective_uv']"
                )
        if self.is_attention_debug_enabled():
            if not isinstance(cross_attention_kwargs, dict) or cross_attention_kwargs.get("sat_perspective_uv") is None:
                raise ValueError(
                    "attention debug requires cross_attention_kwargs['sat_perspective_uv']"
                )
            cross_attention_kwargs = dict(cross_attention_kwargs)
            cross_attention_kwargs["attention_alignment"] = {
                "enabled": True,
                "layers": self._attention_debug_layers,
                "max_query_tokens": None,
                "valid_radius": 0.35,
                "invalid_attention_weight": 0.0,
                "losses": [],
                "metrics": [],
                "debug_storage": self._attention_debug_storage,
            }
            kwargs["cross_attention_kwargs"] = cross_attention_kwargs

        # Diagnostic log (once per process)
        if not SatelliteConditionedUNet._logged_diag:
            _logger = logging.getLogger(__name__)
            _logger.info(
                "[SatelliteConditionedUNet] encoder_hidden_states: "
                "shape=%s mean=%.4f std=%.4f nonzero_frac=%.3f",
                tuple(encoder_hidden_states.shape),
                encoder_hidden_states.float().mean().item(),
                encoder_hidden_states.float().std().item(),
                (encoder_hidden_states.abs() > 1e-6).float().mean().item(),
            )
            SatelliteConditionedUNet._logged_diag = True

        return super().forward(
            *args,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )


def load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Sequence[str], Sequence[str]]:
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    def _is_optional_key(key: str) -> bool:
        return (
            key == "satellite_encoder.perspective_pe_gate"
            or key.startswith("satellite_encoder.perspective_pos_encoder.")
            or ".processor.query_uv_gate" in key
            or ".processor.query_uv_encoder." in key
            or ".processor.geometry_score_gate" in key
            or ".processor.geometry_score_proj." in key
            or ".processor.semantic_query_proj." in key
            or ".processor.semantic_key_proj." in key
        )

    optional_missing = [key for key in missing_keys if _is_optional_key(key)]
    optional_unexpected = [key for key in unexpected_keys if _is_optional_key(key)]
    if optional_missing or optional_unexpected:
        logger.warning(
            "Checkpoint omitted/contained optional geometry-addressing keys; continuing with "
            "initialized geometry params and ignoring removed additive PE params. "
            "missing=%d unexpected=%d",
            len(optional_missing),
            len(optional_unexpected),
        )
    required_missing = [key for key in missing_keys if not _is_optional_key(key)]
    required_unexpected = [key for key in unexpected_keys if not _is_optional_key(key)]
    if required_missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {required_missing}")
    if required_unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {required_unexpected}")
    return missing_keys, unexpected_keys


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    load_model_state_dict(model, state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def _resolve_hf_snapshot_path(model_id: str, revision: Optional[str] = None) -> Optional[Path]:
    candidate_path = Path(model_id).expanduser()
    if candidate_path.is_dir():
        return candidate_path

    if "/" not in model_id:
        return None

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None

    cache_root = Path(hf_home).expanduser() / "hub" / f"models--{model_id.replace('/', '--')}"
    if not cache_root.is_dir():
        return None

    ref_name = revision or "main"
    ref_file = cache_root / "refs" / ref_name
    if ref_file.is_file():
        snapshot_hash = ref_file.read_text().strip()
        snapshot_dir = cache_root / "snapshots" / snapshot_hash
        if snapshot_dir.is_dir():
            return snapshot_dir

    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshot_dirs:
        return None

    return snapshot_dirs[-1]


class SatelliteConditionedSDModel(nn.Module):
    """Stable Diffusion model conditioned by satellite content tokens and geometry addressing."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        noise_scheduler: DDPMScheduler,
        satellite_encoder: Optional[SatelliteConditionEncoder] = None,
        freeze_base: bool = True,
        cond_drop_prob: float = 0.1,
        attention_alignment_enabled: bool = False,
        attention_alignment_loss_weight: float = 0.0,
        attention_alignment_layers: Optional[Sequence[str]] = None,
        attention_alignment_max_query_tokens: Optional[int] = 256,
        attention_alignment_valid_radius: float = 0.25,
        attention_alignment_invalid_attention_weight: float = 0.1,
        transition_aux_enabled: bool = False,
        transition_aux_loss_weight: float = 0.0,
        transition_aux_cycle_weight: float = 0.1,
        transition_aux_composition_weight: float = 0.05,
        transition_aux_mse_weight: float = 0.1,
        transition_aux_hidden_channels: int = 128,
        transition_aux_action_dim: int = 128,
        transition_aux_source: str = TRANSITION_AUX_SOURCE_GT_LATENT,
        joint_view_generation_enabled: bool = False,
        joint_view_generation_loss_weight: float = 0.0,
        joint_view_generation_hidden_dim: int = 32,
        joint_view_generation_num_heads: int = 4,
        joint_view_generation_dropout: float = 0.0,
        joint_view_generation_bev_sigma: float = 0.25,
        joint_view_generation_gate_init: float = 0.0,
    ):
        super().__init__()

        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.cond_drop_prob = float(cond_drop_prob)
        self.perspective_geometry_enabled = True
        self.attention_alignment_enabled = bool(attention_alignment_enabled)
        self.attention_alignment_loss_weight = float(attention_alignment_loss_weight)
        self.attention_alignment_layers = tuple(
            str(layer) for layer in (
                attention_alignment_layers
                if attention_alignment_layers is not None
                else DEFAULT_ATTENTION_ALIGNMENT_LAYERS
            )
        )
        self.attention_alignment_max_query_tokens = (
            None
            if attention_alignment_max_query_tokens is None
            else int(attention_alignment_max_query_tokens)
        )
        self.attention_alignment_valid_radius = float(attention_alignment_valid_radius)
        self.attention_alignment_invalid_attention_weight = float(attention_alignment_invalid_attention_weight)
        self._logged_nondifferentiable_alignment_loss = False
        self.transition_aux_enabled = bool(transition_aux_enabled)
        self.transition_aux_loss_weight = float(transition_aux_loss_weight)
        self.transition_aux_cycle_weight = float(transition_aux_cycle_weight)
        self.transition_aux_composition_weight = float(transition_aux_composition_weight)
        self.transition_aux_mse_weight = float(transition_aux_mse_weight)
        self.transition_aux_source = normalize_transition_aux_source(transition_aux_source)
        self.transition_aux_runtime_scale = 1.0
        self.joint_view_generation_enabled = bool(joint_view_generation_enabled)
        self.joint_view_generation_loss_weight = float(joint_view_generation_loss_weight)
        self.joint_view_generation_runtime_scale = 1.0

        if satellite_encoder is None:
            sat_embed_dim = int(getattr(unet.config, "cross_attention_dim", 768) or 768)
            satellite_encoder = SatelliteConditionEncoder(
                embed_dim=sat_embed_dim,
            )
        self.satellite_encoder = satellite_encoder
        self.transition_head: Optional[TransitionHead] = None
        self.cross_view_refiner: Optional[CrossViewLatentRefiner] = None
        latent_channels = int(
            getattr(
                vae.config,
                "latent_channels",
                getattr(unet.config, "in_channels", 4),
            )
            or 4
        )
        if self.transition_aux_enabled:
            self.transition_head = TransitionHead(
                latent_channels=latent_channels,
                sat_token_dim=int(self.satellite_encoder.embed_dim),
                action_dim=int(transition_aux_action_dim),
                hidden_channels=int(transition_aux_hidden_channels),
            )
        if self.joint_view_generation_enabled:
            self.cross_view_refiner = CrossViewLatentRefiner(
                latent_channels=latent_channels,
                hidden_dim=int(joint_view_generation_hidden_dim),
                num_heads=int(joint_view_generation_num_heads),
                dropout=float(joint_view_generation_dropout),
                bev_sigma=float(joint_view_generation_bev_sigma),
                gate_init=float(joint_view_generation_gate_init),
            )

        if freeze_base:
            for param in self.vae.parameters():
                param.requires_grad = False

            for name, param in self.unet.named_parameters():
                if (
                    ".attn2.to_k." in name
                    or ".attn2.to_v." in name
                    or ".attn2.processor." in name
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for param in self.satellite_encoder.parameters():
            param.requires_grad = True
        if self.cross_view_refiner is not None:
            for param in self.cross_view_refiner.parameters():
                param.requires_grad = True

        logger.info("[SatelliteConditionedSDModel] Initialized")
        logger.info(f"  UNet trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
        logger.info(f"  Satellite encoder params: {sum(p.numel() for p in self.satellite_encoder.parameters())}")
        logger.info(
            f"  Trainable attn2 k/v params: "
            f"{sum(p.numel() for n, p in self.unet.named_parameters() if p.requires_grad and ('.attn2.to_k.' in n or '.attn2.to_v.' in n))}"
        )
        logger.info(
            f"  Trainable attn2 processor params: "
            f"{sum(p.numel() for n, p in self.unet.named_parameters() if p.requires_grad and '.attn2.processor.' in n)}"
        )
        logger.info("  Additive perspective/query PE enabled: False")
        logger.info(f"  Query geometry score enabled: {bool(getattr(self.unet, 'query_geometry_score_enabled', False))}")
        logger.info(f"  Attention alignment enabled: {self.attention_alignment_enabled}")
        logger.info(f"  Attention alignment loss weight: {self.attention_alignment_loss_weight:g}")
        logger.info(f"  Transition auxiliary enabled: {self.transition_aux_enabled}")
        logger.info(f"  Transition auxiliary loss weight: {self.transition_aux_loss_weight:g}")
        logger.info(f"  Transition auxiliary source: {self.transition_aux_source}")
        logger.info(f"  Joint view generation enabled: {self.joint_view_generation_enabled}")
        logger.info(f"  Joint view generation loss weight: {self.joint_view_generation_loss_weight:g}")
        logger.info(f"  Condition dropout: {self.cond_drop_prob}")

    def set_transition_aux_runtime_scale(self, scale: float) -> None:
        self.transition_aux_runtime_scale = float(scale)

    def _effective_transition_aux_weight(self) -> float:
        if (
            not self.training
            or not bool(getattr(self, "transition_aux_enabled", False))
            or getattr(self, "transition_head", None) is None
        ):
            return 0.0
        return float(self.transition_aux_loss_weight) * float(getattr(self, "transition_aux_runtime_scale", 1.0))

    def set_joint_view_generation_runtime_scale(self, scale: float) -> None:
        self.joint_view_generation_runtime_scale = float(scale)

    def _effective_joint_view_generation_weight(self) -> float:
        if (
            not self.training
            or not bool(getattr(self, "joint_view_generation_enabled", False))
            or getattr(self, "cross_view_refiner", None) is None
        ):
            return 0.0
        return float(self.joint_view_generation_loss_weight) * float(
            getattr(self, "joint_view_generation_runtime_scale", 1.0)
        )

    def encode_satellite(
        self,
        sat_images: torch.Tensor,
        *,
        K: Optional[torch.Tensor] = None,
        T_cam_to_world: Optional[torch.Tensor] = None,
        T_imu_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> SatelliteMemoryState:
        """Encode satellite images into a structured memory state."""
        encoder_param = next(self.satellite_encoder.parameters(), None)
        if encoder_param is not None:
            sat_images = sat_images.to(dtype=encoder_param.dtype)
        return self.satellite_encoder(
            sat_images,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            image_size=image_size,
        )

    @staticmethod
    def _normalize_images_for_vae(images: torch.Tensor) -> torch.Tensor:
        return images * 2.0 - 1.0

    def _encode_images_to_latent_mode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_latents = self.vae.encode(self._normalize_images_for_vae(images)).latent_dist
            if hasattr(image_latents, "mode"):
                latents = image_latents.mode()
            else:
                latents = image_latents.sample()
            return latents * self.vae.config.scaling_factor

    @staticmethod
    def _expand_condition_mask(condition_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        mask = condition_mask.to(device=reference.device, dtype=reference.dtype)
        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)
        return mask

    def _sample_condition_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        condition_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        if self.training and self.cond_drop_prob > 0.0:
            condition_mask = torch.rand(batch_size, device=device) >= self.cond_drop_prob
            if not bool(condition_mask.any().item()):
                keep_index = torch.randint(0, batch_size, (1,), device=device)
                condition_mask[keep_index] = True
        return condition_mask

    def _apply_condition_dropout(
        self,
        sat_state: SatelliteMemoryState,
        condition_mask: torch.Tensor,
    ) -> SatelliteMemoryState:
        return sat_state.replace(
            tokens=sat_state.tokens * self._expand_condition_mask(condition_mask, sat_state.tokens),
            xy=sat_state.xy * self._expand_condition_mask(condition_mask, sat_state.xy),
            bev_coords=(
                sat_state.bev_coords * self._expand_condition_mask(condition_mask, sat_state.bev_coords)
                if sat_state.bev_coords is not None
                else None
            ),
            perspective_uv=(
                sat_state.perspective_uv * self._expand_condition_mask(condition_mask, sat_state.perspective_uv)
                if sat_state.perspective_uv is not None
                else None
            ),
            perspective_valid=(
                sat_state.perspective_valid * condition_mask.view(-1, 1)
                if sat_state.perspective_valid is not None
                else None
            ),
        )

    @staticmethod
    def _repeat_satellite_state_for_views(
        sat_state: SatelliteMemoryState,
        num_views: int,
    ) -> SatelliteMemoryState:
        def repeat_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if tensor is None else tensor.repeat_interleave(int(num_views), dim=0)

        return SatelliteMemoryState(
            tokens=sat_state.tokens.repeat_interleave(int(num_views), dim=0),
            xy=sat_state.xy.repeat_interleave(int(num_views), dim=0),
            bev_coords=repeat_optional(sat_state.bev_coords),
            perspective_uv=repeat_optional(sat_state.perspective_uv),
            perspective_valid=repeat_optional(sat_state.perspective_valid),
        )

    @staticmethod
    def _flatten_group_tensor(
        name: str,
        tensor: Optional[torch.Tensor],
        *,
        batch_size: int,
        num_views: int,
        trailing_dims: Tuple[int, ...],
    ) -> torch.Tensor:
        if tensor is None:
            raise ValueError(f"pose-chain group forward requires {name}")
        expected_group_ndim = 2 + len(trailing_dims)
        expected_single_ndim = 1 + len(trailing_dims)
        if tensor.ndim == expected_group_ndim:
            if int(tensor.shape[0]) != batch_size or int(tensor.shape[1]) != num_views:
                raise ValueError(
                    f"{name} must start with [B,V], got {list(tensor.shape)} "
                    f"for B={batch_size}, V={num_views}"
                )
            return tensor.reshape(batch_size * num_views, *trailing_dims)
        if tensor.ndim == expected_single_ndim:
            if int(tensor.shape[0]) != batch_size:
                raise ValueError(
                    f"{name} must start with [B], got {list(tensor.shape)} for B={batch_size}"
                )
            return tensor.repeat_interleave(num_views, dim=0).reshape(batch_size * num_views, *trailing_dims)
        raise ValueError(
            f"{name} must be [B,V{''.join(',' + str(d) for d in trailing_dims)}] "
            f"or [B{''.join(',' + str(d) for d in trailing_dims)}], got {list(tensor.shape)}"
        )

    def _project_group_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        *,
        K: torch.Tensor,
        T_cam_to_world: torch.Tensor,
        T_imu_to_world: torch.Tensor,
        camera_height_m: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> SatelliteMemoryState:
        if sat_state.bev_coords is None:
            raise ValueError("pose-chain group forward requires satellite bev_coords")
        image_h, image_w = int(image_size[0]), int(image_size[1])
        perspective_uv, perspective_valid = compute_sat_patch_perspective_uv(
            bev_coords=sat_state.bev_coords,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            image_w=image_w,
            image_h=image_h,
        )
        return sat_state.replace(
            perspective_uv=perspective_uv,
            perspective_valid=perspective_valid,
        )

    def _build_cross_attention_kwargs(
        self,
        reference: torch.Tensor,
        sat_state: SatelliteMemoryState,
        *,
        chain_group_size: Optional[int] = None,
    ):
        alignment_active = bool(getattr(self, "attention_alignment_enabled", False))
        attention_debug_active = bool(
            hasattr(self.unet, "is_attention_debug_enabled") and self.unet.is_attention_debug_enabled()
        )
        if not bool(getattr(self.unet, "query_geometry_score_enabled", False)) and not alignment_active and not attention_debug_active:
            return None
        if reference.ndim != 4:
            raise ValueError(f"reference tensor must be [B,C,H,W], got {list(reference.shape)}")
        kwargs: Dict[str, Any] = {"query_base_hw": tuple(int(x) for x in reference.shape[-2:])}
        needs_sat_geometry = (
            bool(getattr(self.unet, "query_geometry_score_enabled", False))
            or alignment_active
            or attention_debug_active
        )
        if needs_sat_geometry:
            if sat_state.perspective_uv is None:
                raise ValueError("attention geometry features require sat_state.perspective_uv")
            kwargs["sat_perspective_uv"] = sat_state.perspective_uv
            if sat_state.perspective_valid is not None:
                kwargs["sat_perspective_valid"] = sat_state.perspective_valid
        if alignment_active:
            kwargs["attention_alignment"] = {
                "enabled": True,
                "layers": getattr(self, "attention_alignment_layers", DEFAULT_ATTENTION_ALIGNMENT_LAYERS),
                "max_query_tokens": getattr(self, "attention_alignment_max_query_tokens", 256),
                "valid_radius": getattr(self, "attention_alignment_valid_radius", 0.25),
                "invalid_attention_weight": getattr(self, "attention_alignment_invalid_attention_weight", 0.1),
                "losses": [],
                "metrics": [],
            }
            if chain_group_size is not None:
                kwargs["attention_alignment"]["chain_group_size"] = int(chain_group_size)
        return kwargs

    def _aggregate_attention_alignment(
        self,
        attention_alignment: Optional[Dict[str, Any]],
        *,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = reference.sum() * 0.0
        if not isinstance(attention_alignment, dict):
            return zero, {}

        losses = [
            loss
            for loss in attention_alignment.get("losses", [])
            if torch.is_tensor(loss)
        ]
        if losses:
            alignment_loss = torch.stack([loss.to(device=reference.device).float() for loss in losses]).mean()
            alignment_loss = alignment_loss.to(dtype=reference.dtype)
        else:
            alignment_loss = zero

        metrics: Dict[str, torch.Tensor] = {}
        metric_entries = [
            entry
            for entry in attention_alignment.get("metrics", [])
            if isinstance(entry, dict)
        ]
        for metric_name in (
            "mean_error",
            "valid_query_ratio",
            "valid_attention_mass",
            "valid_attention_mass_without_geometry",
            "target_attention_mass",
            "target_token_fraction",
            "target_attention_lift",
            "target_attention_lift_mixed",
            "target_attention_lift_geometry_only",
            "target_attention_lift_semantic_only",
            "target_attention_mass_without_geometry",
            "target_attention_lift_without_geometry",
            "target_attention_lift_geometry_delta",
            "nearest_attention_mass",
            "nearest_attention_mass_without_geometry",
            "target_logit_gap",
            "target_logit_gap_mixed",
            "target_logit_gap_geometry_only",
            "target_logit_gap_semantic_only",
            "target_logit_gap_without_geometry",
            "target_logit_gap_geometry_delta",
            "content_logits_std",
            "content_logits_abs_mean",
            "content_logits_top_gap",
            "raw_content_qk_std",
            "raw_content_qk_abs_mean",
            "raw_content_qk_top_gap",
            "raw_content_qk_to_geometry_ratio",
            "geometry_bias_std",
            "geometry_bias_abs_mean",
            "geometry_bias_top_gap",
            "geometry_to_content_std_ratio",
            "geometry_to_content_abs_ratio",
            "geometry_to_content_top_gap_ratio",
            "geometry_logits_std",
            "semantic_logits_std",
            "semantic_logits_abs_mean",
            "semantic_logits_top_gap",
            "semantic_to_geometry_ratio",
            "candidate_recall",
            "candidate_count_mean",
            "window_candidate_count_mean",
            "window_fallback_ratio",
            "candidate_valid_query_ratio",
            "attention_geometry_kl",
            "query_content_norm",
            "key_content_norm",
            "geometry_score_gate",
            "geometry_score_gate_raw",
            "geometry_score_runtime_scale",
            "geometry_score_raw_std",
            "geometry_score_bias_std",
            "semantic_score_alpha",
            "semantic_score_raw_std",
            "semantic_score_bias_std",
            "chain_attention_coverage_overlap",
            "chain_attention_centroid_shift",
            "chain_attention_valid_pair_ratio",
        ):
            values = [
                entry[metric_name].to(device=reference.device).float()
                for entry in metric_entries
                if torch.is_tensor(entry.get(metric_name))
            ]
            if values:
                metrics[metric_name] = torch.stack(values).mean().to(dtype=reference.dtype)

        return alignment_loss, metrics

    def _compute_transition_auxiliary(
        self,
        *,
        source_latents: torch.Tensor,
        target_latents: torch.Tensor,
        sat_state: SatelliteMemoryState,
        batch: Dict[str, Any],
        image_hw: Tuple[int, int],
        reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        transition_head = getattr(self, "transition_head", None)
        if not bool(getattr(self, "transition_aux_enabled", False)) or transition_head is None:
            zero = reference * 0.0
            return {
                "transition_aux_loss": zero,
                "transition_aux_weighted_loss": zero,
                "transition_aux_weight": zero,
            }

        outputs = compute_transition_auxiliary_outputs(
            transition_head,
            source_latents=source_latents,
            target_latents=target_latents,
            sat_state=sat_state,
            batch=batch,
            image_hw=image_hw,
            cycle_weight=float(self.transition_aux_cycle_weight),
            composition_weight=float(self.transition_aux_composition_weight),
            mse_weight=float(self.transition_aux_mse_weight),
        )
        weight = torch.tensor(
            float(self._effective_transition_aux_weight()),
            device=reference.device,
            dtype=reference.dtype,
        )
        transition_loss = outputs["loss"].to(device=reference.device, dtype=reference.dtype)
        return {
            "transition_aux_loss": transition_loss,
            "transition_aux_weighted_loss": transition_loss * weight,
            "transition_aux_weight": weight,
            "transition_aux_transition_loss": outputs["transition_loss"].to(device=reference.device, dtype=reference.dtype),
            "transition_aux_front_to_side_loss": outputs["front_to_side_loss"].to(device=reference.device, dtype=reference.dtype),
            "transition_aux_side_to_side_loss": outputs["side_to_side_loss"].to(device=reference.device, dtype=reference.dtype),
            "transition_aux_cycle_loss": outputs["cycle_loss"].to(device=reference.device, dtype=reference.dtype),
            "transition_aux_composition_loss": outputs["composition_loss"].to(device=reference.device, dtype=reference.dtype),
            "transition_aux_num_pairs": outputs["num_pairs"].to(device=reference.device, dtype=reference.dtype),
            "transition_aux_source_predicted_x0": torch.tensor(
                float(self.transition_aux_source == TRANSITION_AUX_SOURCE_PREDICTED_X0),
                device=reference.device,
                dtype=reference.dtype,
            ),
        }

    def _predict_original_sample(
        self,
        *,
        noisy_latents: torch.Tensor,
        model_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = str(getattr(self.noise_scheduler.config, "prediction_type", "epsilon"))
        if prediction_type == "sample":
            return model_pred

        alphas_cumprod = getattr(self.noise_scheduler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            if prediction_type == "epsilon":
                return noisy_latents - model_pred
            raise ValueError(
                f"Cannot compute predicted x0 for prediction_type={prediction_type!r} "
                "because the scheduler does not expose alphas_cumprod"
            )

        alpha_prod_t = alphas_cumprod.to(device=noisy_latents.device, dtype=torch.float32).index_select(
            0,
            timesteps.to(device=noisy_latents.device, dtype=torch.long),
        )
        while alpha_prod_t.ndim < noisy_latents.ndim:
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = (1.0 - alpha_prod_t).clamp_min(0.0)
        sample = noisy_latents.float()
        pred = model_pred.float()

        if prediction_type == "epsilon":
            pred_x0 = (sample - beta_prod_t.sqrt() * pred) / alpha_prod_t.sqrt().clamp_min(1e-8)
        elif prediction_type == "v_prediction":
            pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * pred
        else:
            raise ValueError(
                f"prediction_type must be one of 'epsilon', 'sample', or 'v_prediction', got {prediction_type!r}"
            )
        return pred_x0.to(dtype=model_pred.dtype)

    def _predict_model_output_from_original_sample(
        self,
        *,
        noisy_latents: torch.Tensor,
        pred_x0: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = str(getattr(self.noise_scheduler.config, "prediction_type", "epsilon"))
        if prediction_type == "sample":
            return pred_x0

        alphas_cumprod = getattr(self.noise_scheduler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            if prediction_type == "epsilon":
                return noisy_latents - pred_x0
            raise ValueError(
                f"Cannot convert refined x0 to prediction_type={prediction_type!r} "
                "because the scheduler does not expose alphas_cumprod"
            )

        alpha_prod_t = alphas_cumprod.to(device=noisy_latents.device, dtype=torch.float32).index_select(
            0,
            timesteps.to(device=noisy_latents.device, dtype=torch.long),
        )
        while alpha_prod_t.ndim < noisy_latents.ndim:
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = (1.0 - alpha_prod_t).clamp_min(0.0)
        sample = noisy_latents.float()
        x0 = pred_x0.float()

        if prediction_type == "epsilon":
            model_output = (sample - alpha_prod_t.sqrt() * x0) / beta_prod_t.sqrt().clamp_min(1e-8)
        elif prediction_type == "v_prediction":
            model_output = (alpha_prod_t.sqrt() * sample - x0) / beta_prod_t.sqrt().clamp_min(1e-8)
        else:
            raise ValueError(
                f"prediction_type must be one of 'epsilon', 'sample', or 'v_prediction', got {prediction_type!r}"
            )
        return model_output.to(dtype=pred_x0.dtype)

    def _sample_training_timesteps(
        self,
        *,
        batch_size: int,
        device: torch.device,
        chain_group_size: Optional[int] = None,
    ) -> torch.Tensor:
        if (
            bool(getattr(self, "joint_view_generation_enabled", False))
            and chain_group_size is not None
            and int(chain_group_size) > 1
        ):
            group_size = int(chain_group_size)
            if batch_size % group_size != 0:
                raise ValueError(
                    f"Flattened batch_size={batch_size} must be divisible by chain_group_size={group_size}"
                )
            group_count = batch_size // group_size
            group_timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (group_count,),
                device=device,
                dtype=torch.long,
            )
            return group_timesteps.repeat_interleave(group_size, dim=0)
        return torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

    def _forward_denoising_with_sat_state(
        self,
        target_images: torch.Tensor,
        sat_state: SatelliteMemoryState,
        *,
        condition_mask: Optional[torch.Tensor] = None,
        chain_group_size: Optional[int] = None,
        vehicle_yaw_degs: Optional[torch.Tensor] = None,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = target_images.shape[0]
        if condition_mask is None:
            condition_mask = self._sample_condition_mask(
                batch_size=batch_size,
                device=target_images.device,
            )
        conditioned_sat_state = self._apply_condition_dropout(sat_state, condition_mask)

        with torch.no_grad():
            target_images_vae = self._normalize_images_for_vae(target_images)
            latents = self.vae.encode(target_images_vae).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = self._sample_training_timesteps(
            batch_size=batch_size,
            device=target_images.device,
            chain_group_size=chain_group_size,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        cross_attention_kwargs = self._build_cross_attention_kwargs(
            noisy_latents,
            conditioned_sat_state,
            chain_group_size=chain_group_size,
        )
        attention_alignment = (
            cross_attention_kwargs.get("attention_alignment")
            if isinstance(cross_attention_kwargs, dict)
            else None
        )
        base_model_pred = self.unet(
            noisy_latents,
            timesteps,
            sat_tokens=conditioned_sat_state.tokens,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        base_pred_x0_latents = self._predict_original_sample(
            noisy_latents=noisy_latents,
            model_pred=base_model_pred,
            timesteps=timesteps,
        )
        pred_x0_latents = base_pred_x0_latents
        model_pred = base_model_pred
        joint_consistency_loss = base_model_pred.sum() * 0.0
        joint_weight = torch.tensor(0.0, device=base_model_pred.device, dtype=base_model_pred.dtype)
        joint_metrics: Dict[str, torch.Tensor] = {
            "joint_view_generation_source_refined_x0": torch.tensor(
                0.0, device=base_model_pred.device, dtype=base_model_pred.dtype
            ),
            "joint_view_generation_consistency_loss": joint_consistency_loss,
            "joint_view_generation_weighted_loss": joint_consistency_loss,
            "joint_view_generation_weight": joint_weight,
        }
        cross_view_refiner = getattr(self, "cross_view_refiner", None)
        if (
            bool(getattr(self, "joint_view_generation_enabled", False))
            and cross_view_refiner is not None
            and chain_group_size is not None
            and int(chain_group_size) > 1
        ):
            group_size = int(chain_group_size)
            if batch_size % group_size != 0:
                raise ValueError(
                    f"Flattened batch_size={batch_size} must be divisible by chain_group_size={group_size}"
                )
            group_count = batch_size // group_size
            refiner_output = cross_view_refiner(
                base_pred_x0_latents.reshape(group_count, group_size, *base_pred_x0_latents.shape[1:]),
                target_latents=latents.reshape(group_count, group_size, *latents.shape[1:]),
                vehicle_yaw_degs=vehicle_yaw_degs,
                front_bev_xy=front_bev_xy,
                front_ground_valid_mask=front_ground_valid_mask,
            )
            pred_x0_latents = refiner_output.refined_x0.reshape_as(base_pred_x0_latents)
            model_pred = self._predict_model_output_from_original_sample(
                noisy_latents=noisy_latents,
                pred_x0=pred_x0_latents,
                timesteps=timesteps,
            )
            joint_consistency_loss = refiner_output.consistency_loss.to(device=model_pred.device, dtype=model_pred.dtype)
            joint_weight = torch.tensor(
                float(self._effective_joint_view_generation_weight()),
                device=model_pred.device,
                dtype=model_pred.dtype,
            )
            joint_metrics = {
                "joint_view_generation_source_refined_x0": torch.tensor(
                    1.0, device=model_pred.device, dtype=model_pred.dtype
                ),
                "joint_view_generation_consistency_loss": joint_consistency_loss,
                "joint_view_generation_weighted_loss": joint_consistency_loss * joint_weight,
                "joint_view_generation_weight": joint_weight,
            }
            for key, value in refiner_output.metrics.items():
                normalized_key = key.replace("/", "_").replace("=", "_")
                joint_metrics[normalized_key] = value.to(device=model_pred.device, dtype=model_pred.dtype)

        per_item_denoise_loss = (model_pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
        denoise_loss = per_item_denoise_loss.mean().to(dtype=model_pred.dtype)
        alignment_loss, alignment_metrics = self._aggregate_attention_alignment(
            attention_alignment,
            reference=denoise_loss,
        )
        captured_alignment_losses = (
            attention_alignment.get("losses", [])
            if isinstance(attention_alignment, dict)
            else []
        )
        alignment_requested = (
            isinstance(attention_alignment, dict)
            and bool(attention_alignment.get("enabled", False))
        )
        alignment_loss_is_differentiable = any(
            torch.is_tensor(loss_value) and bool(loss_value.requires_grad)
            for loss_value in captured_alignment_losses
        )
        effective_alignment_weight = float(getattr(self, "attention_alignment_loss_weight", 0.0))
        if not alignment_requested or not self.training:
            effective_alignment_weight = 0.0
        elif effective_alignment_weight > 0.0 and not alignment_loss_is_differentiable:
            effective_alignment_weight = 0.0
            if not getattr(self, "_logged_nondifferentiable_alignment_loss", False):
                logger.warning(
                    "Attention alignment loss is being logged but not added to the objective because "
                    "the captured tensors are non-differentiable. This usually happens with UNet gradient "
                    "checkpointing; disable gradient_checkpointing or use a smaller batch to train with this loss."
                )
                self._logged_nondifferentiable_alignment_loss = True
        loss = (
            denoise_loss
            + float(effective_alignment_weight) * alignment_loss
            + joint_metrics["joint_view_generation_weighted_loss"]
        )
        result = {
            "loss": loss,
            "denoise_loss": denoise_loss,
            "per_item_denoise_loss": per_item_denoise_loss.to(dtype=loss.dtype),
            "attention_alignment_loss": alignment_loss,
            "attention_alignment_loss_weight": torch.tensor(
                float(effective_alignment_weight),
                device=loss.device,
                dtype=loss.dtype,
            ),
            "attention_alignment_loss_is_differentiable": torch.tensor(
                float(alignment_loss_is_differentiable),
                device=loss.device,
                dtype=loss.dtype,
            ),
            "model_pred": model_pred,
            "base_model_pred": base_model_pred,
            "base_pred_x0_latents": base_pred_x0_latents,
            "pred_x0_latents": pred_x0_latents,
            "target": target,
            "timesteps": timesteps,
            "sat_state": conditioned_sat_state,
            "condition_mask": condition_mask,
        }
        result.update(joint_metrics)
        for key, value in alignment_metrics.items():
            result[f"attention_alignment_{key}"] = value
        return result

    @staticmethod
    def _zero_sat_geometry(sat_state: SatelliteMemoryState) -> SatelliteMemoryState:
        return sat_state.replace(
            perspective_uv=(
                torch.zeros_like(sat_state.perspective_uv)
                if sat_state.perspective_uv is not None
                else None
            ),
            perspective_valid=(
                torch.zeros_like(sat_state.perspective_valid)
                if sat_state.perspective_valid is not None
                else None
            ),
        )

    @staticmethod
    def _concat_optional_tensors(left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if left is None or right is None:
            return None
        return torch.cat([left, right], dim=0)

    def _build_cfg_sat_state(self, sat_state: SatelliteMemoryState) -> SatelliteMemoryState:
        zero_tokens = torch.zeros_like(sat_state.tokens)
        zero_geometry = self._zero_sat_geometry(sat_state)
        return SatelliteMemoryState(
            tokens=torch.cat([sat_state.tokens, zero_tokens], dim=0),
            xy=torch.cat([sat_state.xy, sat_state.xy], dim=0),
            bev_coords=self._concat_optional_tensors(sat_state.bev_coords, sat_state.bev_coords),
            perspective_uv=self._concat_optional_tensors(sat_state.perspective_uv, zero_geometry.perspective_uv),
            perspective_valid=self._concat_optional_tensors(sat_state.perspective_valid, zero_geometry.perspective_valid),
        )

    @torch.no_grad()
    def _get_vae_scale_factor(self) -> int:
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels is None:
            return 8
        return max(1, 2 ** (len(block_out_channels) - 1))

    @staticmethod
    def _infer_generation_size(
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if target_size is None:
            raise ValueError("target_size is required when satellite conditioning is used")
        return int(target_size[0]), int(target_size[1])

    @torch.no_grad()
    def generate_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        target_size: Optional[Tuple[int, int]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sat_condition_mode: str = "normal",
    ) -> Tuple[torch.Tensor, SatelliteMemoryState]:
        """Generate images from a precomputed satellite memory state."""
        B = sat_state.tokens.shape[0]
        device = sat_state.tokens.device

        if sat_condition_mode == "normal":
            condition_mask = torch.ones(B, device=device, dtype=torch.bool)
        elif sat_condition_mode == "zero":
            sat_state = sat_state.replace(
                tokens=torch.zeros_like(sat_state.tokens),
                perspective_uv=(
                    torch.zeros_like(sat_state.perspective_uv)
                    if sat_state.perspective_uv is not None
                    else None
                ),
                perspective_valid=(
                    torch.zeros_like(sat_state.perspective_valid)
                    if sat_state.perspective_valid is not None
                    else None
                ),
            )
            condition_mask = torch.zeros(B, device=device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown sat_condition_mode: {sat_condition_mode}")

        image_h, image_w = self._infer_generation_size(target_size=target_size)
        vae_scale_factor = self._get_vae_scale_factor()
        latent_h = max(1, (image_h + vae_scale_factor - 1) // vae_scale_factor)
        latent_w = max(1, (image_w + vae_scale_factor - 1) // vae_scale_factor)
        unet_param = next(self.unet.parameters(), None)
        latent_dtype = unet_param.dtype if unet_param is not None else sat_state.tokens.dtype

        latents = torch.randn(
            (B, self.unet.config.in_channels, latent_h, latent_w),
            device=device,
            dtype=latent_dtype,
            generator=generator,
        )

        use_cfg = guidance_scale > 1.0
        if use_cfg:
            sat_state_double = self._build_cfg_sat_state(sat_state)

        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            cross_attention_kwargs = self._build_cross_attention_kwargs(
                latents,
                sat_state_double if use_cfg else sat_state,
            )
            if use_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                noise_pred_both = self.unet(
                    latent_model_input,
                    t,
                    sat_tokens=sat_state_double.tokens,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                noise_pred_cond, noise_pred_uncond = noise_pred_both.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(
                    latents,
                    t,
                    sat_tokens=sat_state.tokens,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            if generator is not None:
                try:
                    latents = self.noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
                except TypeError:
                    latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            else:
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        vae_param = next(self.vae.parameters(), None)
        vae_dtype = vae_param.dtype if vae_param is not None else latents.dtype
        latents = (latents / self.vae.config.scaling_factor).to(dtype=vae_dtype)
        generated_images = self.vae.decode(latents).sample
        generated_images = (generated_images / 2 + 0.5).clamp(0, 1)

        if generated_images.shape[-2:] != (image_h, image_w):
            generated_images = F.interpolate(
                generated_images,
                size=(image_h, image_w),
                mode="bilinear",
                align_corners=False,
            )

        return generated_images, sat_state

    @torch.no_grad()
    def generate(
        self,
        sat_images: torch.Tensor,
        *,
        K: Optional[torch.Tensor] = None,
        T_cam_to_world: Optional[torch.Tensor] = None,
        T_imu_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sat_condition_mode: str = "normal",
    ) -> torch.Tensor:
        sat_state = self.encode_satellite(
            sat_images,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            image_size=target_size,
        )
        generated_images, _ = self.generate_with_satellite_state(
            sat_state,
            target_size=target_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            sat_condition_mode=sat_condition_mode,
        )
        return generated_images

    @torch.no_grad()
    def generate_pose_chain(
        self,
        sat_images: torch.Tensor,
        *,
        K: torch.Tensor,
        T_cam_to_world: torch.Tensor,
        T_imu_to_world: torch.Tensor,
        camera_height_m: torch.Tensor,
        vehicle_yaw_degs: Optional[torch.Tensor] = None,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sat_condition_mode: str = "normal",
    ) -> torch.Tensor:
        """Jointly sample a fixed pose chain with the cross-view refiner."""
        if K.ndim != 4:
            raise ValueError(f"generate_pose_chain expects grouped K [B,V,3,3], got {list(K.shape)}")
        batch_size, num_views = int(K.shape[0]), int(K.shape[1])
        if int(sat_images.shape[0]) != batch_size:
            raise ValueError(f"sat_images batch {sat_images.shape[0]} does not match K batch {batch_size}")

        image_h, image_w = self._infer_generation_size(target_size=target_size)
        target_size = (image_h, image_w)
        base_sat_state = self.encode_satellite(sat_images, image_size=None)
        view_sat_state = self._repeat_satellite_state_for_views(base_sat_state, num_views=num_views)
        view_sat_state = self._project_group_satellite_state(
            view_sat_state,
            K=self._flatten_group_tensor("K", K, batch_size=batch_size, num_views=num_views, trailing_dims=(3, 3)),
            T_cam_to_world=self._flatten_group_tensor(
                "T_cam_to_world",
                T_cam_to_world,
                batch_size=batch_size,
                num_views=num_views,
                trailing_dims=(4, 4),
            ),
            T_imu_to_world=self._flatten_group_tensor(
                "T_imu_to_world",
                T_imu_to_world,
                batch_size=batch_size,
                num_views=num_views,
                trailing_dims=(4, 4),
            ),
            camera_height_m=self._flatten_group_tensor(
                "camera_height_m",
                camera_height_m,
                batch_size=batch_size,
                num_views=num_views,
                trailing_dims=(),
            ),
            image_size=target_size,
        )

        flat_batch = batch_size * num_views
        device = view_sat_state.tokens.device
        if sat_condition_mode == "normal":
            condition_mask = torch.ones(flat_batch, device=device, dtype=torch.bool)
        elif sat_condition_mode == "zero":
            condition_mask = torch.zeros(flat_batch, device=device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown sat_condition_mode: {sat_condition_mode}")
        view_sat_state = self._apply_condition_dropout(view_sat_state, condition_mask)

        vae_scale_factor = self._get_vae_scale_factor()
        latent_h = max(1, (image_h + vae_scale_factor - 1) // vae_scale_factor)
        latent_w = max(1, (image_w + vae_scale_factor - 1) // vae_scale_factor)
        unet_param = next(self.unet.parameters(), None)
        latent_dtype = unet_param.dtype if unet_param is not None else view_sat_state.tokens.dtype

        latents = torch.randn(
            (flat_batch, self.unet.config.in_channels, latent_h, latent_w),
            device=device,
            dtype=latent_dtype,
            generator=generator,
        )

        use_cfg = guidance_scale > 1.0
        if use_cfg:
            cfg_sat_state = self._build_cfg_sat_state(view_sat_state)

        self.noise_scheduler.set_timesteps(num_inference_steps)
        cross_view_refiner = getattr(self, "cross_view_refiner", None)

        for t in self.noise_scheduler.timesteps:
            if torch.is_tensor(t):
                timestep_tensor = t.to(device=device, dtype=torch.long).reshape(1).expand(flat_batch)
            else:
                timestep_tensor = torch.full((flat_batch,), int(t), device=device, dtype=torch.long)

            if use_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                cfg_timestep = torch.cat([timestep_tensor, timestep_tensor], dim=0)
                cross_attention_kwargs = self._build_cross_attention_kwargs(
                    latent_model_input,
                    cfg_sat_state,
                    chain_group_size=num_views,
                )
                model_pred_both = self.unet(
                    latent_model_input,
                    cfg_timestep,
                    sat_tokens=cfg_sat_state.tokens,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                model_pred_cond, model_pred_uncond = model_pred_both.chunk(2, dim=0)
                model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
            else:
                cross_attention_kwargs = self._build_cross_attention_kwargs(
                    latents,
                    view_sat_state,
                    chain_group_size=num_views,
                )
                model_pred = self.unet(
                    latents,
                    timestep_tensor,
                    sat_tokens=view_sat_state.tokens,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            if (
                bool(getattr(self, "joint_view_generation_enabled", False))
                and cross_view_refiner is not None
                and num_views > 1
            ):
                pred_x0 = self._predict_original_sample(
                    noisy_latents=latents,
                    model_pred=model_pred,
                    timesteps=timestep_tensor,
                )
                refiner_output = cross_view_refiner(
                    pred_x0.reshape(batch_size, num_views, *pred_x0.shape[1:]),
                    vehicle_yaw_degs=vehicle_yaw_degs,
                    front_bev_xy=front_bev_xy,
                    front_ground_valid_mask=front_ground_valid_mask,
                )
                model_pred = self._predict_model_output_from_original_sample(
                    noisy_latents=latents,
                    pred_x0=refiner_output.refined_x0.reshape_as(pred_x0),
                    timesteps=timestep_tensor,
                )

            if generator is not None:
                try:
                    latents = self.noise_scheduler.step(model_pred, t, latents, generator=generator).prev_sample
                except TypeError:
                    latents = self.noise_scheduler.step(model_pred, t, latents).prev_sample
            else:
                latents = self.noise_scheduler.step(model_pred, t, latents).prev_sample

        vae_param = next(self.vae.parameters(), None)
        vae_dtype = vae_param.dtype if vae_param is not None else latents.dtype
        decode_latents = (latents / self.vae.config.scaling_factor).to(dtype=vae_dtype)
        generated_images = self.vae.decode(decode_latents).sample
        generated_images = (generated_images / 2 + 0.5).clamp(0, 1)

        if generated_images.shape[-2:] != target_size:
            generated_images = F.interpolate(
                generated_images,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return generated_images.reshape(batch_size, num_views, *generated_images.shape[1:])

    def forward(
        self,
        sat_images: torch.Tensor,
        target_images: torch.Tensor,
        *,
        K: Optional[torch.Tensor] = None,
        T_cam_to_world: Optional[torch.Tensor] = None,
        T_imu_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
        vehicle_yaw_degs: Optional[torch.Tensor] = None,
        view_names: Optional[Any] = None,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if target_images.ndim == 5:
            batch_size, num_views = int(target_images.shape[0]), int(target_images.shape[1])
            flat_target_images = target_images.reshape(batch_size * num_views, *target_images.shape[2:])
            target_size = tuple(int(x) for x in flat_target_images.shape[-2:])

            base_sat_state = self.encode_satellite(
                sat_images,
                image_size=None,
            )
            view_sat_state = self._repeat_satellite_state_for_views(
                base_sat_state,
                num_views=num_views,
            )
            flat_K = self._flatten_group_tensor(
                "K",
                K,
                batch_size=batch_size,
                num_views=num_views,
                trailing_dims=(3, 3),
            )
            flat_T_cam_to_world = self._flatten_group_tensor(
                "T_cam_to_world",
                T_cam_to_world,
                batch_size=batch_size,
                num_views=num_views,
                trailing_dims=(4, 4),
            )
            flat_T_imu_to_world = self._flatten_group_tensor(
                "T_imu_to_world",
                T_imu_to_world,
                batch_size=batch_size,
                num_views=num_views,
                trailing_dims=(4, 4),
            )
            flat_camera_height_m = self._flatten_group_tensor(
                "camera_height_m",
                camera_height_m,
                batch_size=batch_size,
                num_views=num_views,
                trailing_dims=(),
            )
            view_sat_state = self._project_group_satellite_state(
                view_sat_state,
                K=flat_K,
                T_cam_to_world=flat_T_cam_to_world,
                T_imu_to_world=flat_T_imu_to_world,
                camera_height_m=flat_camera_height_m,
                image_size=target_size,
            )
            group_condition_mask = self._sample_condition_mask(
                batch_size=batch_size,
                device=target_images.device,
            ).repeat_interleave(num_views, dim=0)
            denoising_kwargs: Dict[str, Any] = {}
            if bool(getattr(self, "joint_view_generation_enabled", False)):
                denoising_kwargs.update(
                    {
                        "vehicle_yaw_degs": vehicle_yaw_degs,
                        "front_bev_xy": front_bev_xy,
                        "front_ground_valid_mask": front_ground_valid_mask,
                    }
                )
            result = self._forward_denoising_with_sat_state(
                flat_target_images,
                view_sat_state,
                condition_mask=group_condition_mask,
                chain_group_size=num_views,
                **denoising_kwargs,
            )
            if bool(getattr(self, "transition_aux_enabled", False)) and getattr(self, "transition_head", None) is not None:
                transition_target_latents = self._encode_images_to_latent_mode(flat_target_images)
                if self.transition_aux_source == TRANSITION_AUX_SOURCE_PREDICTED_X0:
                    transition_source_latents = result["pred_x0_latents"]
                else:
                    transition_source_latents = transition_target_latents
                transition_batch: Dict[str, Any] = {
                    "K": K,
                    "T_cam_to_world": T_cam_to_world,
                    "T_imu_to_world": T_imu_to_world,
                    "camera_height_m": camera_height_m,
                }
                if vehicle_yaw_degs is not None:
                    transition_batch["vehicle_yaw_degs"] = vehicle_yaw_degs
                if view_names is not None:
                    transition_batch["view_names"] = view_names
                transition_outputs = self._compute_transition_auxiliary(
                    source_latents=transition_source_latents,
                    target_latents=transition_target_latents,
                    sat_state=base_sat_state,
                    batch=transition_batch,
                    image_hw=target_size,
                    reference=result["loss"],
                )
                result["loss"] = result["loss"] + transition_outputs["transition_aux_weighted_loss"]
                result.update(transition_outputs)
            result["chain_group_size"] = torch.tensor(
                float(num_views),
                device=result["loss"].device,
                dtype=result["loss"].dtype,
            )
            result["per_view_denoise_loss"] = result["per_item_denoise_loss"].reshape(
                batch_size,
                num_views,
            )
            result["chain_denoise_loss"] = result["per_view_denoise_loss"].mean(dim=1)
            return result

        if target_images.ndim != 4:
            raise ValueError(
                "target_images must be [B,C,H,W] for single-view training or "
                f"[B,V,C,H,W] for pose-chain group training, got {list(target_images.shape)}"
            )

        sat_state = self.encode_satellite(
            sat_images,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            image_size=tuple(int(x) for x in target_images.shape[-2:]),
        )

        return self._forward_denoising_with_sat_state(target_images, sat_state)


def create_sd_model(
    base_model: str = "stabilityai/stable-diffusion-2-1-base",
    freeze_base: bool = True,
    revision: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cond_drop_prob: float = 0.1,
    perspective_pe_enabled: bool = False,
    query_uv_pe_enabled: bool = False,
    query_geometry_bias_enabled: bool = False,
    query_geometry_bias_scale: float = 2.0,
    query_geometry_invalid_penalty: float = -1e4,
    query_geometry_score_enabled: bool = False,
    query_geometry_score_dim: int = 64,
    query_geometry_score_num_freqs: int = 6,
    query_geometry_score_gate_init: float = 1.0,
    query_geometry_score_layers: Optional[Sequence[str]] = None,
    query_geometry_score_max_query_tokens: Optional[int] = None,
    query_geometry_score_mode: str = "geometry_first_semantic_refine",
    query_geometry_candidate_radius: float = 0.35,
    query_geometry_candidate_min_k: int = 16,
    query_geometry_candidate_invalid_penalty: float = -1e4,
    query_semantic_score_dim: int = 64,
    query_semantic_score_alpha: float = 0.25,
    query_uv_gate_init: float = 0.0,
    attention_alignment_enabled: bool = False,
    attention_alignment_loss_weight: float = 0.0,
    attention_alignment_layers: Optional[Sequence[str]] = None,
    attention_alignment_max_query_tokens: Optional[int] = 256,
    attention_alignment_valid_radius: float = 0.25,
    attention_alignment_invalid_attention_weight: float = 0.1,
    transition_aux_enabled: bool = False,
    transition_aux_loss_weight: float = 0.0,
    transition_aux_cycle_weight: float = 0.1,
    transition_aux_composition_weight: float = 0.05,
    transition_aux_mse_weight: float = 0.1,
    transition_aux_hidden_channels: int = 128,
    transition_aux_action_dim: int = 128,
    transition_aux_source: str = TRANSITION_AUX_SOURCE_GT_LATENT,
    joint_view_generation_enabled: bool = False,
    joint_view_generation_loss_weight: float = 0.0,
    joint_view_generation_hidden_dim: int = 32,
    joint_view_generation_num_heads: int = 4,
    joint_view_generation_dropout: float = 0.0,
    joint_view_generation_bev_sigma: float = 0.25,
    joint_view_generation_gate_init: float = 0.0,
    satellite_encoder_config: Optional[Dict[str, Any]] = None,
) -> SatelliteConditionedSDModel:
    """Create a satellite-conditioned Stable Diffusion model."""
    del perspective_pe_enabled, query_uv_pe_enabled, query_geometry_bias_enabled
    del query_geometry_bias_scale, query_geometry_invalid_penalty, query_uv_gate_init
    resolved_base_model = _resolve_hf_snapshot_path(base_model, revision=revision)
    load_source = str(resolved_base_model) if resolved_base_model is not None else base_model
    if resolved_base_model is not None:
        logger.info(f"Using cached base model snapshot: {resolved_base_model}")

    component_load_kwargs: Dict[str, Any] = {}
    if revision is not None and resolved_base_model is None:
        component_load_kwargs["revision"] = revision
    if torch_dtype is not None:
        component_load_kwargs["torch_dtype"] = torch_dtype
    if resolved_base_model is not None:
        component_load_kwargs["local_files_only"] = True

    vae = AutoencoderKL.from_pretrained(
        load_source,
        subfolder="vae",
        **component_load_kwargs,
    )
    base_unet = UNet2DConditionModel.from_pretrained(
        load_source,
        subfolder="unet",
        **component_load_kwargs,
    )

    sat_encoder_cfg = dict(satellite_encoder_config or {})
    sat_encoder_cfg.pop("name", None)
    deprecated_sat_keys = {
        "perspective_pe_enabled",
        "perspective_num_freqs",
        "perspective_pe_gate_init",
        "perspective_invalid_mode",
        "perspective_use_validity_embedding",
        "perspective_ooi_init_std",
        "perspective_validity_embed_init_std",
        "perspective_pe_injection",
        "perspective_pe_scale_mode",
        "perspective_pe_target_ratio",
    }
    removed_sat_keys = sorted(key for key in deprecated_sat_keys if key in sat_encoder_cfg)
    for key in removed_sat_keys:
        sat_encoder_cfg.pop(key, None)
    if removed_sat_keys:
        logger.warning(
            "Ignoring removed additive satellite PE config keys: %s",
            ", ".join(removed_sat_keys),
        )
    sat_embed_dim = int(getattr(base_unet.config, "cross_attention_dim", 768) or 768)
    requested_embed_dim = sat_encoder_cfg.get("embed_dim")
    if requested_embed_dim is not None and int(requested_embed_dim) != sat_embed_dim:
        logger.warning(
            "Overriding satellite_encoder.embed_dim=%s with base UNet cross_attention_dim=%s",
            requested_embed_dim,
            sat_embed_dim,
        )
    sat_encoder_cfg["embed_dim"] = sat_embed_dim
    satellite_encoder = SatelliteConditionEncoder(**sat_encoder_cfg)

    unet = SatelliteConditionedUNet(
        query_geometry_score_enabled=query_geometry_score_enabled,
        query_geometry_score_dim=query_geometry_score_dim,
        query_geometry_score_num_freqs=query_geometry_score_num_freqs,
        query_geometry_score_gate_init=query_geometry_score_gate_init,
        query_geometry_score_layers=query_geometry_score_layers,
        query_geometry_score_max_query_tokens=query_geometry_score_max_query_tokens,
        query_geometry_score_mode=query_geometry_score_mode,
        query_geometry_candidate_radius=query_geometry_candidate_radius,
        query_geometry_candidate_min_k=query_geometry_candidate_min_k,
        query_geometry_candidate_invalid_penalty=query_geometry_candidate_invalid_penalty,
        query_semantic_score_dim=query_semantic_score_dim,
        query_semantic_score_alpha=query_semantic_score_alpha,
        attention_alignment_enabled=attention_alignment_enabled,
        **base_unet.config,
    )
    unet.load_state_dict(base_unet.state_dict(), strict=False)
    if torch_dtype is not None:
        unet = unet.to(dtype=torch_dtype)

    scheduler_load_kwargs: Dict[str, Any] = {}
    if revision is not None and resolved_base_model is None:
        scheduler_load_kwargs["revision"] = revision
    if resolved_base_model is not None:
        scheduler_load_kwargs["local_files_only"] = True

    noise_scheduler = DDPMScheduler.from_pretrained(
        load_source,
        subfolder="scheduler",
        **scheduler_load_kwargs,
    )

    model = SatelliteConditionedSDModel(
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        satellite_encoder=satellite_encoder,
        freeze_base=freeze_base,
        cond_drop_prob=cond_drop_prob,
        attention_alignment_enabled=attention_alignment_enabled,
        attention_alignment_loss_weight=attention_alignment_loss_weight,
        attention_alignment_layers=attention_alignment_layers,
        attention_alignment_max_query_tokens=attention_alignment_max_query_tokens,
        attention_alignment_valid_radius=attention_alignment_valid_radius,
        attention_alignment_invalid_attention_weight=attention_alignment_invalid_attention_weight,
        transition_aux_enabled=transition_aux_enabled,
        transition_aux_loss_weight=transition_aux_loss_weight,
        transition_aux_cycle_weight=transition_aux_cycle_weight,
        transition_aux_composition_weight=transition_aux_composition_weight,
        transition_aux_mse_weight=transition_aux_mse_weight,
        transition_aux_hidden_channels=transition_aux_hidden_channels,
        transition_aux_action_dim=transition_aux_action_dim,
        transition_aux_source=transition_aux_source,
        joint_view_generation_enabled=joint_view_generation_enabled,
        joint_view_generation_loss_weight=joint_view_generation_loss_weight,
        joint_view_generation_hidden_dim=joint_view_generation_hidden_dim,
        joint_view_generation_num_heads=joint_view_generation_num_heads,
        joint_view_generation_dropout=joint_view_generation_dropout,
        joint_view_generation_bev_sigma=joint_view_generation_bev_sigma,
        joint_view_generation_gate_init=joint_view_generation_gate_init,
    )
    return model
