"""
Stable Diffusion Trainer for satellite-to-frontview generation.

This module provides a simplified training interface using diffusers library.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence, Tuple
import logging
from PIL import Image, ImageDraw, ImageOps

from models.conditioning import SatelliteMemoryState


logger = logging.getLogger(__name__)


DEFAULT_ATTENTION_VIS_LAYERS = [
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2",
    "mid_block.attentions.0.transformer_blocks.0.attn2",
]

def load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Sequence[str], Sequence[str]]:
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
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



def create_sd_model(
    base_model: str = 'stabilityai/stable-diffusion-2-1-base',
    freeze_base: bool = True,
    revision: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cond_drop_prob: float = 0.1,
    perspective_pe_enabled: bool = True,
    query_uv_pe_enabled: bool = False,
    query_geometry_bias_enabled: bool = False,
    query_geometry_bias_scale: float = 2.0,
    query_geometry_invalid_penalty: float = -1e4,
    query_uv_gate_init: float = 0.0,
    attention_alignment_enabled: bool = False,
    attention_alignment_loss_weight: float = 0.0,
    attention_alignment_layers: Optional[Sequence[str]] = None,
    attention_alignment_max_query_tokens: Optional[int] = 256,
    attention_alignment_valid_radius: float = 0.25,
    attention_alignment_invalid_attention_weight: float = 0.1,
    satellite_encoder_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Backward-compatible import surface for the clean perspective-PE model."""
    from models.sd_model import create_sd_model as _create_sd_model

    return _create_sd_model(
        base_model=base_model,
        freeze_base=freeze_base,
        revision=revision,
        torch_dtype=torch_dtype,
        cond_drop_prob=cond_drop_prob,
        perspective_pe_enabled=perspective_pe_enabled,
        query_uv_pe_enabled=query_uv_pe_enabled,
        query_geometry_bias_enabled=query_geometry_bias_enabled,
        query_geometry_bias_scale=query_geometry_bias_scale,
        query_geometry_invalid_penalty=query_geometry_invalid_penalty,
        query_uv_gate_init=query_uv_gate_init,
        attention_alignment_enabled=attention_alignment_enabled,
        attention_alignment_loss_weight=attention_alignment_loss_weight,
        attention_alignment_layers=attention_alignment_layers,
        attention_alignment_max_query_tokens=attention_alignment_max_query_tokens,
        attention_alignment_valid_radius=attention_alignment_valid_radius,
        attention_alignment_invalid_attention_weight=attention_alignment_invalid_attention_weight,
        satellite_encoder_config=satellite_encoder_config,
    )


from models.sd_model import SatelliteConditionedSDModel as SatelliteConditionedSDModel


class SDTrainer:
    """
    Trainer for satellite-to-frontview generation with Stable Diffusion.

    Handles training loop, optimization, checkpointing, and evaluation.
    """

    def __init__(
        self,
        model: SatelliteConditionedSDModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_train_epochs: int = 100,
        lr_scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
        gradient_accumulation_steps: int = 1,
        output_dir: str = './output',
        save_every: int = 5,
        log_every: int = 100,
        device: str = 'cuda',
        use_wandb: bool = False,
        project_name: str = 'kitti360_sd',
        wandb_run_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_mode: str = 'online',
        use_tensorboard: bool = False,
        tensorboard_log_dir: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        mixed_precision: Optional[str] = None,
        max_grad_norm: float = 1.0,
        visualize_every: int = 1,
        num_visualizations: int = 4,
        visualization_inference_steps: int = 20,
        visualization_guidance_scale: float = 1.0,
        visualization_seed: int = 42,
        attention_visualization_layers: Optional[Sequence[str]] = None,
        distributed: bool = False,
        local_rank: int = 0,
    ):
        self.model = model.to(device)
        self.distributed = bool(distributed)
        self.local_rank = int(local_rank)
        self.rank = dist.get_rank() if self.distributed and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if self.distributed and dist.is_initialized() else 1
        self.is_main_process = self.rank == 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_epochs = warmup_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = Path(output_dir)
        self.save_every = save_every
        self.log_every = log_every
        self.device = device
        self.use_wandb = bool(use_wandb) and self.is_main_process
        self.project_name = project_name
        self.wandb_run_name = wandb_run_name
        self.wandb_entity = wandb_entity
        self.wandb_mode = wandb_mode
        self.use_tensorboard = use_tensorboard
        self.max_grad_norm = float(max_grad_norm)
        self.visualize_every = max(0, int(visualize_every))
        self.num_visualizations = max(0, int(num_visualizations))
        self.visualization_inference_steps = int(visualization_inference_steps)
        self.visualization_guidance_scale = float(visualization_guidance_scale)
        self.visualization_seed = int(visualization_seed)
        self.attention_visualization_layers = list(
            attention_visualization_layers
            if attention_visualization_layers is not None
            else DEFAULT_ATTENTION_VIS_LAYERS
        )
        self.visualization_dir = self.output_dir / "visualizations"
        if self.is_main_process:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_log_dir = Path(tensorboard_log_dir) if tensorboard_log_dir is not None else self.output_dir / "tensorboard"
        self.tb_writer = None
        self.mixed_precision = None if mixed_precision is None else mixed_precision.lower()
        self.use_amp = device.startswith("cuda") and self.mixed_precision in {"fp16", "bf16"}
        self.amp_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16
            if self.mixed_precision == "bf16"
            else None
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        # Setup output dir
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        self._materialize_lazy_condition_modules()
        self._ensure_trainable_params_fp32()
        self._assert_no_trainable_fp16_params()
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.startswith("cuda") else None,
                output_device=self.local_rank if self.device.startswith("cuda") else None,
                find_unused_parameters=True,
            )

        # Setup optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup scheduler
        num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / gradient_accumulation_steps))
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        warmup_steps = warmup_epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Setup wandb
        if use_wandb and self.is_main_process:
            if not is_wandb_available():
                raise ImportError("Please install wandb to use logging: pip install wandb")
            import wandb
            wandb.init(
                project=project_name,
                entity=wandb_entity,
                name=wandb_run_name,
                mode=wandb_mode,
                config=run_config,
            )
        if use_tensorboard and self.is_main_process:
            try:
                from tensorboardX import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "Please install tensorboardX to use TensorBoard logging: pip install tensorboardX"
                ) from exc
            self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(logdir=str(self.tensorboard_log_dir))

        if self.is_main_process:
            logger.info(f"[SDTrainer] Initialized")
            logger.info(f"  Distributed: {self.distributed} (world_size={self.world_size})")
            logger.info(f"  Learning rate: {learning_rate}")
            logger.info(f"  Num epochs: {num_train_epochs}")
            logger.info(f"  Batch size per process: {train_dataloader.batch_size}")
            logger.info(
                "  Effective batch size: %d",
                int(train_dataloader.batch_size or 0) * self.world_size * self.gradient_accumulation_steps,
            )
            logger.info(f"  Mixed precision: {self.mixed_precision or 'disabled'}")
            logger.info(f"  Max grad norm: {self.max_grad_norm}")
            unwrapped = self.unwrapped_model
            unet = getattr(unwrapped, "unet", None)
            logger.info(f"  Perspective PE enabled: {bool(getattr(unwrapped, 'perspective_pe_enabled', False))}")
            logger.info(f"  Query UV PE enabled: {bool(getattr(unet, 'query_uv_pe_enabled', False))}")
            logger.info(f"  Query UV gate init: {float(getattr(unet, 'query_uv_gate_init', 0.0))}")
            logger.info(f"  Query geometry bias enabled: {bool(getattr(unet, 'query_geometry_bias_enabled', False))}")
            logger.info(f"  Query geometry score enabled: {bool(getattr(unet, 'query_geometry_score_enabled', False))}")
            logger.info(f"  Attention alignment enabled: {bool(getattr(unwrapped, 'attention_alignment_enabled', False))}")
            logger.info(
                "  Attention alignment loss weight: %.6g",
                float(getattr(unwrapped, "attention_alignment_loss_weight", 0.0)),
            )
            if self.use_wandb:
                logger.info(
                    f"  W&B logging: project={self.project_name}, "
                    f"run_name={self.wandb_run_name or 'auto'}, mode={self.wandb_mode}"
                )
            if self.tb_writer is not None:
                logger.info(f"  TensorBoard log dir: {self.tensorboard_log_dir}")
            if self.visualize_every > 0 and self.num_visualizations > 0:
                logger.info(
                    f"  Visualization: every {self.visualize_every} epoch(s), "
                    f"{self.num_visualizations} sample(s), {self.visualization_inference_steps} denoise steps"
                )

    @property
    def unwrapped_model(self) -> SatelliteConditionedSDModel:
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _reduce_mean(self, value: float, device: str) -> float:
        if not self.distributed or not dist.is_initialized():
            return float(value)
        tensor = torch.tensor(float(value), device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(self.world_size)
        return float(tensor.item())

    def _any_rank_true(self, value: bool) -> bool:
        if not self.distributed or not dist.is_initialized():
            return bool(value)
        device = torch.device(self.device)
        flag = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _move_batch_geometry(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        geometry: Dict[str, torch.Tensor] = {}
        for key in ("K", "T_cam_to_world", "T_imu_to_world", "camera_height_m"):
            value = batch.get(key)
            if torch.is_tensor(value):
                geometry[key] = value.to(self.device)
        return geometry

    @torch.no_grad()
    def _materialize_lazy_condition_modules(self) -> None:
        return

    def _ensure_trainable_params_fp32(self) -> None:
        fp16_params = []
        converted_param_count = 0
        converted_numel = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.dtype != torch.float16:
                continue
            fp16_params.append(name)
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()
            converted_param_count += 1
            converted_numel += param.numel()

        if fp16_params:
            logger.info(
                f"Converted {converted_param_count} trainable parameter tensors "
                f"({converted_numel} values) from fp16 to fp32 for AMP stability"
            )

    def _assert_no_trainable_fp16_params(self) -> None:
        remaining_fp16 = [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.dtype == torch.float16
        ]
        if remaining_fp16:
            preview = ", ".join(remaining_fp16[:8])
            if len(remaining_fp16) > 8:
                preview += ", ..."
            raise RuntimeError(
                "Found trainable fp16 parameters after AMP preparation. "
                f"These must stay fp32 for GradScaler: {preview}"
            )

    @staticmethod
    def _tensor_debug_summary(name: str, tensor: torch.Tensor) -> str:
        with torch.no_grad():
            detached = tensor.detach()
            finite_mask = torch.isfinite(detached)
            finite_count = int(finite_mask.sum().item())
            total_count = detached.numel()
            summary = (
                f"{name}: shape={list(detached.shape)} dtype={detached.dtype} "
                f"finite={finite_count}/{total_count}"
            )
            if finite_count > 0:
                finite_values = detached[finite_mask].float()
                summary += (
                    f" min={finite_values.min().item():.4g}"
                    f" max={finite_values.max().item():.4g}"
                    f" mean={finite_values.mean().item():.4g}"
                )
            return summary

    @staticmethod
    def _first_nonfinite_named_tensor(
        named_tensors: Sequence[Tuple[str, Optional[torch.Tensor]]],
    ) -> Optional[str]:
        for name, tensor in named_tensors:
            if tensor is None or not torch.is_tensor(tensor):
                continue
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                return SDTrainer._tensor_debug_summary(name, tensor)
        return None

    @staticmethod
    def _grad_global_norm(parameters: Sequence[torch.nn.Parameter], device: str) -> torch.Tensor:
        norms = []
        for param in parameters:
            if param.grad is None:
                continue
            norms.append(torch.linalg.vector_norm(param.grad.detach().float()))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.linalg.vector_norm(torch.stack([norm.to(device=device) for norm in norms]))

    def _log_nonfinite_training_state(
        self,
        reason: str,
        epoch: int,
        step: int,
        outputs: Optional[Dict[str, Any]] = None,
        batch_tensors: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> None:
        if not self.is_main_process:
            return

        logger.warning(
            "Skipping optimizer update because non-finite values were detected "
            "(reason=%s, epoch=%d, step=%d)",
            reason,
            epoch + 1,
            step + 1,
        )

        if outputs:
            output_tensors: List[Tuple[str, Optional[torch.Tensor]]] = []
            for key in ("loss", "per_view_loss", "model_pred", "target"):
                value = outputs.get(key)
                if torch.is_tensor(value):
                    output_tensors.append((f"outputs.{key}", value))
            sat_state = outputs.get("sat_state")
            if isinstance(sat_state, SatelliteMemoryState):
                output_tensors.extend(
                    [
                        ("outputs.sat_state.tokens", sat_state.tokens),
                        ("outputs.sat_state.xy", sat_state.xy),
                        ("outputs.sat_state.bev_coords", sat_state.bev_coords),
                        ("outputs.sat_state.perspective_uv", sat_state.perspective_uv),
                        ("outputs.sat_state.perspective_valid", sat_state.perspective_valid),
                    ]
                )
            issue = self._first_nonfinite_named_tensor(output_tensors)
            if issue is not None:
                logger.warning("  First non-finite output: %s", issue)

        if batch_tensors:
            issue = self._first_nonfinite_named_tensor(
                [(f"batch.{key}", value) for key, value in batch_tensors.items()]
            )
            if issue is not None:
                logger.warning("  First non-finite batch tensor: %s", issue)

        param_issue = self._first_nonfinite_named_tensor(
            [(name, param) for name, param in self.unwrapped_model.named_parameters()]
        )
        if param_issue is not None:
            logger.warning("  First non-finite parameter: %s", param_issue)

    def _global_step(self, epoch: int, step: int) -> int:
        return epoch * max(1, len(self.train_dataloader)) + step + 1

    @staticmethod
    def _should_log_train_step(step: int, num_batches: int, log_every: int) -> bool:
        current_step = int(step) + 1
        if current_step <= 0 or int(num_batches) <= 0:
            return False
        if current_step == int(num_batches):
            return True
        return int(log_every) > 0 and current_step % int(log_every) == 0

    def _log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        scalar_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if value is not None
        }
        if not scalar_metrics:
            return

        if self.use_wandb:
            import wandb
            wandb.log(scalar_metrics, step=step)

        if self.tb_writer is not None:
            for key, value in scalar_metrics.items():
                self.tb_writer.add_scalar(key, value, global_step=step)

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1)

    def _log_visualizations(
        self,
        images: Sequence[Image.Image],
        captions: Sequence[str],
        step: int,
    ) -> None:
        if not images:
            return

        if self.use_wandb:
            import wandb
            wandb.log(
                {
                    "visualizations": [
                        wandb.Image(image, caption=caption)
                        for image, caption in zip(images, captions)
                    ]
                },
                step=step,
            )

        if self.tb_writer is not None:
            for idx, (image, caption) in enumerate(zip(images, captions)):
                tag = f"visualizations/sample_{idx:02d}"
                self.tb_writer.add_image(
                    tag,
                    self._pil_to_tensor(image),
                    global_step=step,
                    dataformats="CHW",
                )
                self.tb_writer.add_text(
                    f"{tag}_caption",
                    caption,
                    global_step=step,
                )

    def _close_loggers(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.use_wandb:
            import wandb
            wandb.finish()

    def train(self, resume_from: Optional[str] = None):
        """Run training."""
        start_epoch = 0

        try:
            if resume_from is not None:
                start_epoch = self._load_checkpoint(resume_from)
            if start_epoch >= self.num_train_epochs:
                logger.info(
                    "Checkpoint already reached configured training horizon: "
                    "start_epoch=%d num_train_epochs=%d",
                    start_epoch,
                    self.num_train_epochs,
                )
                return
            for epoch in range(start_epoch, self.num_train_epochs):
                if self.is_main_process:
                    logger.info(f"Epoch {epoch + 1}/{self.num_train_epochs}")
                train_raw_loss = self._train_epoch(epoch)
                epoch_step = self._global_step(epoch, max(0, len(self.train_dataloader) - 1))
                if self.is_main_process:
                    logger.info(f"  Train raw loss: {train_raw_loss:.4f}")
                    self._log_scalars(
                        {
                            "train/epoch_raw_loss": train_raw_loss,
                            "train/epoch": epoch + 1,
                        },
                        step=epoch_step,
                    )

                # Validate
                if self.val_dataloader is not None and self.is_main_process:
                    val_loss = self._validate(epoch)
                    logger.info(f"  Val loss: {val_loss:.4f}")
                    self._log_scalars(
                        {
                            "val/loss": val_loss,
                            "train/epoch": epoch + 1,
                        },
                        step=epoch_step,
                    )
                self._barrier()

                if self.is_main_process and self.visualize_every > 0 and self.num_visualizations > 0 and (epoch + 1) % self.visualize_every == 0:
                    self._save_visualizations(epoch)
                self._barrier()

                # Save checkpoint
                if self.is_main_process and ((epoch + 1) % self.save_every == 0 or (epoch + 1) == self.num_train_epochs):
                    self._save_checkpoint(epoch)
                self._barrier()
        finally:
            self._close_loggers()

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_raw_loss = 0.0
        finite_loss_batches = 0
        num_batches = len(self.train_dataloader)
        self.optimizer.zero_grad(set_to_none=True)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        skip_accumulation_until = -1
        last_grad_norm = None

        sampler = getattr(self.train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        dataset = getattr(self.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Train Epoch {epoch+1}",
            disable=not self.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            accumulation_start = (step // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
            accumulation_end = min(accumulation_start + self.gradient_accumulation_steps, num_batches)
            accumulation_window_size = max(1, accumulation_end - accumulation_start)
            if step < skip_accumulation_until:
                if self.is_main_process:
                    progress_bar.set_postfix({"raw_loss": "skipped"})
                if (step + 1) == skip_accumulation_until:
                    skip_accumulation_until = -1
                continue

            # Move data to device
            sat_images = batch['sat'].to(self.device)
            target_images = batch['image'].to(self.device)
            geometry_kwargs = self._move_batch_geometry(batch)

            # Forward pass
            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    sat_images,
                    target_images,
                    **geometry_kwargs,
                )
                raw_loss = outputs['loss']

            batch_tensors = {
                "sat": sat_images,
                "image": target_images,
                **geometry_kwargs,
            }
            loss_is_finite = bool(torch.isfinite(raw_loss.detach()).all().item())
            if self._any_rank_true(not loss_is_finite):
                self._log_nonfinite_training_state(
                    reason="loss",
                    epoch=epoch,
                    step=step,
                    outputs=outputs,
                    batch_tensors=batch_tensors,
                )
                if self.is_main_process:
                    self._log_scalars(
                        {
                            "train/skipped_nonfinite_update": 1.0,
                            "train/epoch": epoch + 1,
                        },
                        step=self._global_step(epoch, step),
                    )
                self.optimizer.zero_grad(set_to_none=True)
                skip_accumulation_until = accumulation_end
                if (step + 1) == accumulation_end:
                    skip_accumulation_until = -1
                continue

            # Backward pass
            loss = raw_loss / accumulation_window_size
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) == accumulation_end:
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            self.max_grad_norm,
                        )
                    else:
                        grad_norm = self._grad_global_norm(trainable_params, self.device)
                else:
                    if self.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            self.max_grad_norm,
                        )
                    else:
                        grad_norm = self._grad_global_norm(trainable_params, self.device)

                last_grad_norm = float(grad_norm.detach().float().item())
                grad_is_finite = bool(torch.isfinite(grad_norm.detach()).all().item())
                if self._any_rank_true(not grad_is_finite):
                    self._log_nonfinite_training_state(
                        reason="grad",
                        epoch=epoch,
                        step=step,
                        outputs=outputs,
                        batch_tensors=batch_tensors,
                    )
                    if self.is_main_process:
                        self._log_scalars(
                            {
                                "train/skipped_nonfinite_update": 1.0,
                                "train/grad_norm": last_grad_norm,
                                "train/epoch": epoch + 1,
                            },
                            step=self._global_step(epoch, step),
                        )
                    if self.use_amp and self.amp_dtype == torch.float16:
                        self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_raw_loss += raw_loss.item()
            finite_loss_batches += 1
            postfix = {'raw_loss': f"{raw_loss.item():.3f}"}
            if torch.is_tensor(outputs.get("attention_alignment_mean_error")):
                postfix["attn_err"] = f"{outputs['attention_alignment_mean_error'].detach().float().item():.3f}"
            if self.is_main_process:
                progress_bar.set_postfix(postfix)

            # Log
            if self.is_main_process and self._should_log_train_step(step, num_batches, self.log_every):
                logger.info(
                    "Train step %d/%d: raw_loss=%.6f",
                    step + 1,
                    num_batches,
                    raw_loss.item(),
                )
                log_payload = {
                    'train/raw_loss': raw_loss.item(),
                    'train/lr': self.lr_scheduler.get_last_lr()[0],
                    'train/epoch': epoch + 1,
                }
                metric_name_map = {
                    "denoise_loss": "train/denoise_loss",
                    "attention_alignment_loss": "train/attention_alignment/loss",
                    "attention_alignment_mean_error": "train/attention_alignment/mean_error",
                    "attention_alignment_valid_query_ratio": "train/attention_alignment/valid_query_ratio",
                    "attention_alignment_valid_attention_mass": "train/attention_alignment/valid_attention_mass",
                    "attention_alignment_target_attention_mass": "train/attention_alignment/target_attention_mass",
                    "attention_alignment_target_token_fraction": "train/attention_alignment/target_token_fraction",
                    "attention_alignment_target_attention_lift": "train/attention_alignment/target_attention_lift",
                    "attention_alignment_nearest_attention_mass": "train/attention_alignment/nearest_attention_mass",
                    "attention_alignment_target_logit_gap": "train/attention_alignment/target_logit_gap",
                    "attention_alignment_query_content_norm": "train/attention_alignment/query_content_norm",
                    "attention_alignment_query_pe_norm": "train/attention_alignment/query_pe_norm",
                    "attention_alignment_query_pe_ratio": "train/attention_alignment/query_pe_ratio",
                    "attention_alignment_query_uv_gate": "train/attention_alignment/query_uv_gate",
                    "attention_alignment_key_content_norm": "train/attention_alignment/key_content_norm",
                    "attention_alignment_geometry_score_gate": "train/attention_alignment/geometry_score_gate",
                    "attention_alignment_geometry_score_raw_std": "train/attention_alignment/geometry_score_raw_std",
                    "attention_alignment_geometry_score_bias_std": "train/attention_alignment/geometry_score_bias_std",
                    "attention_alignment_loss_weight": "train/attention_alignment/loss_weight",
                    "attention_alignment_loss_is_differentiable": "train/attention_alignment/loss_is_differentiable",
                }
                for output_key, metric_key in metric_name_map.items():
                    value = outputs.get(output_key)
                    if torch.is_tensor(value):
                        log_payload[metric_key] = float(value.detach().float().item())
                if last_grad_norm is not None:
                    log_payload['train/grad_norm'] = last_grad_norm
                self._log_scalars(log_payload, step=self._global_step(epoch, step))

        local_mean = total_raw_loss / max(1, finite_loss_batches)
        return self._reduce_mean(local_mean, self.device)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Validate the model."""
        eval_model = self.unwrapped_model
        was_training = eval_model.training
        eval_model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_dataloader, desc=f"Val Epoch {epoch+1}"):
            sat_images = batch['sat'].to(self.device)
            target_images = batch['image'].to(self.device)
            geometry_kwargs = self._move_batch_geometry(batch)

            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = eval_model(
                    sat_images,
                    target_images,
                    **geometry_kwargs,
                )
                loss = outputs['loss']
            total_loss += loss.item()

        if was_training:
            eval_model.train()
        return total_loss / len(self.val_dataloader)

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }

        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
        logger.info(f"Checkpoint saved: {checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'}")

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        image = image.detach().cpu().clamp(0, 1)
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        image = (image * 255).to(torch.uint8).numpy()
        return Image.fromarray(image)

    @staticmethod
    def _coords_to_satellite_pixels(
        coords: torch.Tensor,
        sat_width: int,
        sat_height: int,
    ) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
        if coords.numel() == 0:
            return [], None

        valid = (
            torch.isfinite(coords).all(dim=-1)
            & ~((coords[:, 0].abs() < 1e-6) & (coords[:, 1].abs() < 1e-6))
            & (coords[:, 0] >= -1.0)
            & (coords[:, 0] <= 1.0)
            & (coords[:, 1] >= -1.0)
            & (coords[:, 1] <= 1.0)
        )
        coords = coords[valid]
        if coords.numel() == 0:
            return [], None

        x_px = (coords[:, 0] + 1.0) * 0.5 * float(max(1, sat_width - 1))
        y_px = (1.0 - (coords[:, 1] + 1.0) * 0.5) * float(max(1, sat_height - 1))
        points = list(zip(x_px.tolist(), y_px.tolist()))
        bbox = (
            float(x_px.min().item()),
            float(y_px.min().item()),
            float(x_px.max().item()),
            float(y_px.max().item()),
        )
        return points, bbox

    @classmethod
    def _front_bev_xy_to_satellite_pixels(
        cls,
        front_bev_xy: Optional[torch.Tensor],
        sat_width: int,
        sat_height: int,
    ) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
        if front_bev_xy is None or not torch.is_tensor(front_bev_xy):
            return [], None

        coords = front_bev_xy.detach().cpu().to(torch.float32)
        if coords.ndim == 3 and coords.shape[0] == 2:
            coords = coords.permute(1, 2, 0).reshape(-1, 2)
        elif coords.ndim == 3 and coords.shape[-1] == 2:
            coords = coords.reshape(-1, 2)
        elif coords.ndim == 2 and coords.shape[-1] == 2:
            pass
        else:
            return [], None

        return cls._coords_to_satellite_pixels(coords, sat_width, sat_height)

    @staticmethod
    def _convex_hull_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(points) <= 3:
            return points

        unique_points = sorted(set((float(x), float(y)) for x, y in points))
        if len(unique_points) <= 3:
            return unique_points

        def cross(
            origin: Tuple[float, float],
            a: Tuple[float, float],
            b: Tuple[float, float],
        ) -> float:
            return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

        lower: List[Tuple[float, float]] = []
        for point in unique_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper: List[Tuple[float, float]] = []
        for point in reversed(unique_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        return lower[:-1] + upper[:-1]

    @staticmethod
    def _front_mask_to_hw(front_ground_valid_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if front_ground_valid_mask is None or not torch.is_tensor(front_ground_valid_mask):
            return None

        mask = front_ground_valid_mask.detach().cpu().to(torch.float32)
        if mask.ndim == 3 and mask.shape[0] == 1:
            return mask[0] > 0.5
        if mask.ndim == 3 and mask.shape[-1] == 1:
            return mask[..., 0] > 0.5
        if mask.ndim == 2:
            return mask > 0.5
        return None

    @classmethod
    def _front_bev_xy_to_fov_polygon(
        cls,
        front_bev_xy: Optional[torch.Tensor],
        sat_width: int,
        sat_height: int,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float]]:
        if front_bev_xy is None or not torch.is_tensor(front_bev_xy):
            return []

        coords = front_bev_xy.detach().cpu().to(torch.float32)
        if coords.ndim == 3 and coords.shape[0] == 2:
            coords_hw = coords.permute(1, 2, 0)
        elif coords.ndim == 3 and coords.shape[-1] == 2:
            coords_hw = coords
        else:
            return []

        height, width = int(coords_hw.shape[0]), int(coords_hw.shape[1])
        if height < 2 or width < 2:
            return []

        valid_mask = cls._front_mask_to_hw(front_ground_valid_mask)
        if valid_mask is not None and tuple(valid_mask.shape) == (height, width):
            valid_coords = coords_hw[valid_mask]
            points, bbox = cls._coords_to_satellite_pixels(valid_coords, sat_width, sat_height)
            if len(points) >= 3:
                max_hull_points = 6000
                if len(points) > max_hull_points:
                    step = max(1, len(points) // max_hull_points)
                    points = points[::step]
                hull = cls._convex_hull_points(points)
                if len(hull) >= 3:
                    return hull
            if bbox is not None:
                left_px, top_px, right_px, bottom_px = bbox
                return [
                    (left_px, top_px),
                    (right_px, top_px),
                    (right_px, bottom_px),
                    (left_px, bottom_px),
                ]

        top = coords_hw[0, :, :]
        right = coords_hw[:, width - 1, :]
        bottom = torch.flip(coords_hw[height - 1, :, :], dims=[0])
        left = torch.flip(coords_hw[:, 0, :], dims=[0])
        boundary = torch.cat([top, right, bottom, left], dim=0)
        points, _ = cls._coords_to_satellite_pixels(boundary, sat_width, sat_height)

        if len(points) < 3:
            _, bbox = cls._front_bev_xy_to_satellite_pixels(coords_hw, sat_width, sat_height)
            if bbox is None:
                return []
            left_px, top_px, right_px, bottom_px = bbox
            return [
                (left_px, top_px),
                (right_px, top_px),
                (right_px, bottom_px),
                (left_px, bottom_px),
            ]
        return points

    @classmethod
    def _draw_satellite_view_coverage(
        cls,
        sat_image: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor],
        front_ground_valid_mask: Optional[torch.Tensor],
        view_label: Optional[str],
        yaw_deg: Optional[float],
    ) -> Image.Image:
        image = cls._tensor_to_pil(sat_image).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        polygon = cls._front_bev_xy_to_fov_polygon(
            front_bev_xy,
            width,
            height,
            front_ground_valid_mask=front_ground_valid_mask,
        )

        center = (width / 2.0, height / 2.0)
        cross = max(4, int(round(min(width, height) * 0.027)))
        draw.line((center[0] - cross, center[1], center[0] + cross, center[1]), fill=(255, 255, 255, 230), width=2)
        draw.line((center[0], center[1] - cross, center[0], center[1] + cross), fill=(255, 255, 255, 230), width=2)
        draw.ellipse((center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3), fill=(255, 64, 64, 240))

        if polygon:
            draw.polygon(polygon, fill=(0, 180, 255, 45), outline=(255, 230, 0, 235))
            draw.line(polygon + [polygon[0]], fill=(255, 230, 0, 235), width=2)

        if view_label or yaw_deg is not None:
            label = str(view_label) if view_label else "view"
            if yaw_deg is not None:
                label = f"{label} yaw={float(yaw_deg):g}"
            draw.rectangle((4, 4, min(width - 4, 210), 25), fill=(0, 0, 0, 150))
            draw.text((8, 8), label, fill=(255, 255, 255, 255))
        return image

    def _compose_visualization(
        self,
        sat_image: torch.Tensor,
        generated_image: torch.Tensor,
        real_image: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        view_label: Optional[str] = None,
        yaw_deg: Optional[float] = None,
    ) -> Image.Image:
        target_h, target_w = int(real_image.shape[-2]), int(real_image.shape[-1])
        sat_resized = F.interpolate(
            sat_image.unsqueeze(0),
            size=(target_h, target_h),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        sat_pil = self._draw_satellite_view_coverage(
            sat_resized,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
            view_label=view_label,
            yaw_deg=yaw_deg,
        )
        gen_pil = self._tensor_to_pil(generated_image)
        real_pil = self._tensor_to_pil(real_image)

        canvas = Image.new("RGB", (sat_pil.width + gen_pil.width + real_pil.width, target_h))
        x_offset = 0
        for img in (sat_pil, gen_pil, real_pil):
            canvas.paste(img, (x_offset, 0))
            x_offset += img.width
        return canvas

    @staticmethod
    def _safe_filename_token(value: str) -> str:
        return (
            str(value)
            .replace("/", "_")
            .replace(".", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _normalize_heatmap(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        values = values.detach().float()
        finite = torch.isfinite(values)
        if not bool(finite.any().item()):
            return torch.zeros_like(values, dtype=torch.float32)
        min_value = values[finite].min()
        max_value = values[finite].max()
        normalized = (values - min_value) / (max_value - min_value + eps)
        return normalized.clamp(0, 1).masked_fill(~finite, 0)

    @classmethod
    def _heatmap_to_pil(cls, values: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        normalized = cls._normalize_heatmap(values)
        image = (normalized * 255).to(torch.uint8).cpu().numpy()
        pil = Image.fromarray(image, mode="L")
        pil = ImageOps.colorize(pil, black=(0, 0, 60), white=(255, 48, 0), mid=(255, 220, 0))
        if size is not None and pil.size != size:
            pil = pil.resize(size, Image.Resampling.BILINEAR)
        return pil.convert("RGB")

    @staticmethod
    def _xy_to_pixel(xy: torch.Tensor, width: int, height: int) -> Tuple[float, float]:
        x = (float(xy[0]) + 1.0) * 0.5 * float(max(1, width - 1))
        y = (1.0 - (float(xy[1]) + 1.0) * 0.5) * float(max(1, height - 1))
        return x, y

    @classmethod
    def _draw_target_marker(
        cls,
        image: Image.Image,
        xy: torch.Tensor,
        *,
        color: Tuple[int, int, int] = (0, 255, 255),
    ) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        x, y = cls._xy_to_pixel(xy, width, height)
        radius = max(3, int(round(min(width, height) * 0.018)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color + (255,), width=2)
        draw.line((x - radius * 1.6, y, x + radius * 1.6, y), fill=color + (255,), width=2)
        draw.line((x, y - radius * 1.6, x, y + radius * 1.6), fill=color + (255,), width=2)

    @staticmethod
    def _select_attention_query_indices(
        query_mask: Optional[torch.Tensor],
        query_hw: Tuple[int, int],
    ) -> List[int]:
        height, width = int(query_hw[0]), int(query_hw[1])
        candidates = [
            (int(round(height * 0.78)), int(round(width * 0.50))),
            (int(round(height * 0.66)), int(round(width * 0.30))),
            (int(round(height * 0.66)), int(round(width * 0.70))),
            (int(round(height * 0.50)), int(round(width * 0.50))),
        ]
        indices = []
        flat_mask = query_mask.reshape(-1).bool() if torch.is_tensor(query_mask) else None
        for row, col in candidates:
            row = min(max(row, 0), height - 1)
            col = min(max(col, 0), width - 1)
            index = row * width + col
            if flat_mask is not None and not bool(flat_mask[index].item()):
                valid = torch.nonzero(flat_mask, as_tuple=False).flatten()
                if valid.numel() == 0:
                    continue
                grid = torch.stack(
                    [valid // width, valid % width],
                    dim=-1,
                ).float()
                target = torch.tensor([row, col], dtype=torch.float32)
                index = int(valid[((grid - target).pow(2).sum(dim=-1)).argmin()].item())
            if index not in indices:
                indices.append(index)
        return indices

    def _save_attention_debug_visualizations(
        self,
        *,
        attention_debug: Dict[str, Any],
        sat_image: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor],
        front_ground_valid_mask: Optional[torch.Tensor],
        output_dir: Path,
        prefix: str,
    ) -> None:
        if not attention_debug:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        sat_base = self._tensor_to_pil(sat_image).convert("RGB")
        sat_width, sat_height = sat_base.size

        for layer_name, payload in attention_debug.items():
            attention = payload.get("attention")
            front_xy = payload.get("front_xy")
            sat_xy = payload.get("sat_xy")
            query_hw = payload.get("query_hw")
            sat_hw = payload.get("sat_hw")
            if (
                not torch.is_tensor(attention)
                or not torch.is_tensor(front_xy)
                or not torch.is_tensor(sat_xy)
                or query_hw is None
            ):
                continue
            if attention.ndim != 3 or front_xy.ndim != 3 or sat_xy.ndim != 3:
                continue

            attention = attention[0]
            front_xy = front_xy[0]
            sat_xy = sat_xy[0]
            query_mask = payload.get("query_mask")
            if torch.is_tensor(query_mask):
                query_mask = query_mask[0]

            if sat_hw is None:
                side = int(math.isqrt(attention.shape[-1]))
                sat_hw = (side, side) if side * side == attention.shape[-1] else None
            if sat_hw is None:
                continue

            layer_token = self._safe_filename_token(layer_name)
            query_indices = self._select_attention_query_indices(query_mask, query_hw)
            heatmaps = []
            for query_index in query_indices:
                heat = attention[query_index].reshape(int(sat_hw[0]), int(sat_hw[1]))
                heat_pil = self._heatmap_to_pil(heat, size=(sat_width, sat_height))
                overlay = Image.blend(sat_base, heat_pil, alpha=0.55)
                self._draw_target_marker(overlay, front_xy[query_index], color=(0, 255, 255))
                draw = ImageDraw.Draw(overlay, "RGBA")
                row = int(query_index) // int(query_hw[1])
                col = int(query_index) % int(query_hw[1])
                draw.rectangle((4, 4, 150, 25), fill=(0, 0, 0, 150))
                draw.text((8, 8), f"q=({row},{col})", fill=(255, 255, 255, 255))
                heatmaps.append(overlay)
                overlay.save(output_dir / f"{prefix}_{layer_token}_q{query_index:04d}_sat_heatmap.png")

            if heatmaps:
                canvas = Image.new("RGB", (sat_width * len(heatmaps), sat_height))
                for idx, image in enumerate(heatmaps):
                    canvas.paste(image, (idx * sat_width, 0))
                canvas.save(output_dir / f"{prefix}_{layer_token}_sat_heatmaps.png")

            predicted_xy = torch.matmul(attention, sat_xy.float())
            error = (predicted_xy - front_xy.float()).pow(2).sum(dim=-1).sqrt()
            if torch.is_tensor(query_mask):
                valid_query = query_mask.reshape(-1).bool()
                attention_for_error = attention.masked_fill(~valid_query[:, None], 0.0)
                error_for_error = error.masked_fill(~valid_query, 0.0)
            else:
                attention_for_error = attention
                error_for_error = error

            sat_error_weight = attention_for_error.sum(dim=0)
            sat_error = (attention_for_error * error_for_error[:, None]).sum(dim=0)
            sat_error = sat_error / sat_error_weight.clamp_min(1e-8)
            sat_error = sat_error.masked_fill(sat_error_weight <= 1e-8, float("nan"))
            bev_error_map = sat_error.reshape(int(sat_hw[0]), int(sat_hw[1]))
            bev_error_pil = self._heatmap_to_pil(bev_error_map, size=(sat_width, sat_height))

            base = sat_base.copy()
            if front_bev_xy is not None:
                draw = ImageDraw.Draw(base, "RGBA")
                polygon = self._front_bev_xy_to_fov_polygon(
                    front_bev_xy,
                    sat_width,
                    sat_height,
                    front_ground_valid_mask=front_ground_valid_mask,
                )
                if polygon:
                    draw.polygon(polygon, fill=(0, 180, 255, 35), outline=(255, 230, 0, 220))
                    draw.line(polygon + [polygon[0]], fill=(255, 230, 0, 220), width=2)
            bev_error_pil = Image.blend(base, bev_error_pil, alpha=0.55)
            bev_error_pil.save(output_dir / f"{prefix}_{layer_token}_bev_alignment_error.png")

    @staticmethod
    def _visualization_view_specs() -> List[Tuple[str, Optional[float]]]:
        return [
            ("front", None),
            ("yaw_m120", -120.0),
            ("yaw_m90", -90.0),
            ("yaw_m60", -60.0),
            ("yaw_p60", 60.0),
            ("yaw_p90", 90.0),
            ("yaw_p120", 120.0),
        ]

    @staticmethod
    def _sample_with_visualization_view(base_sample: Any, view_label: str, yaw_deg: Optional[float]) -> Any:
        base_meta = getattr(base_sample, "meta", None)
        meta = dict(base_meta) if isinstance(base_meta, dict) else {}
        meta["view_name"] = view_label
        if yaw_deg is None:
            meta["mode_override"] = "front"
            meta.pop("vehicle_relative_yaw_deg_override", None)
            meta.pop("fisheye_relative_yaw_deg_override", None)
        else:
            meta["mode_override"] = "fisheye_virtual"
            meta["vehicle_relative_yaw_deg_override"] = float(yaw_deg)

        return type(base_sample)(
            drive_dir=getattr(base_sample, "drive_dir"),
            frame_id=getattr(base_sample, "frame_id"),
            meta=meta,
        )

    def _collect_fixed_yaw_visualization_items(self, data_loader: DataLoader) -> Optional[List[Dict[str, Any]]]:
        dataset = getattr(data_loader, "dataset", None)
        samples = getattr(dataset, "samples", None)
        if dataset is None or not samples:
            return None

        base_index = 0
        base_sample = samples[base_index]
        items = []
        for view_label, yaw_deg in self._visualization_view_specs():
            override_sample = self._sample_with_visualization_view(base_sample, view_label, yaw_deg)
            original_sample = samples[base_index]
            try:
                samples[base_index] = override_sample
                item = dataset[base_index]
            finally:
                samples[base_index] = original_sample

            if not isinstance(item, dict):
                return None
            item = dict(item)
            item["_visualization_view_label"] = view_label
            item["_visualization_yaw_deg"] = yaw_deg
            items.append(item)

        return items

    @staticmethod
    def _stack_visualization_items(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not items:
            return None

        required_keys = ("sat", "image", "K", "T_cam_to_world", "T_imu_to_world")
        for key in required_keys:
            if any(key not in item or not torch.is_tensor(item[key]) for item in items):
                return None

        frame_ids = []
        for item in items:
            frame_id = item.get("frame_id")
            if torch.is_tensor(frame_id):
                frame_id = int(frame_id.item())
            frame_ids.append(frame_id)

        result = {
            "sat": torch.stack([item["sat"] for item in items], dim=0),
            "image": torch.stack([item["image"] for item in items], dim=0),
            "frame_ids": frame_ids,
            "view_labels": [str(item["_visualization_view_label"]) for item in items],
            "yaw_degs": [item["_visualization_yaw_deg"] for item in items],
        }
        camera_heights = []
        for item in items:
            camera_height = item.get("camera_height_m")
            if torch.is_tensor(camera_height):
                camera_height = float(camera_height.item())
            if camera_height is None:
                return None
            camera_heights.append(float(camera_height))
        result["camera_height_m"] = torch.tensor(camera_heights, dtype=torch.float32)

        if all("front_bev_xy" in item and torch.is_tensor(item["front_bev_xy"]) for item in items):
            result["front_bev_xy"] = torch.stack([item["front_bev_xy"] for item in items], dim=0)
        else:
            result["front_bev_xy"] = None
        if all(
            "front_ground_valid_mask" in item and torch.is_tensor(item["front_ground_valid_mask"])
            for item in items
        ):
            result["front_ground_valid_mask"] = torch.stack(
                [item["front_ground_valid_mask"] for item in items],
                dim=0,
            )
        else:
            result["front_ground_valid_mask"] = None
        for cam_key in ("K", "T_cam_to_world", "T_imu_to_world"):
            if all(cam_key in item and torch.is_tensor(item[cam_key]) for item in items):
                result[cam_key] = torch.stack([item[cam_key] for item in items], dim=0)
        return result

    def _collect_fallback_visualization_batch(self, data_loader: DataLoader) -> Optional[Dict[str, Any]]:
        sat_chunks = []
        target_chunks = []
        front_bev_xy_chunks = []
        front_ground_valid_mask_chunks = []
        K_chunks = []
        T_cam_to_world_chunks = []
        T_imu_to_world_chunks = []
        camera_height_chunks = []
        frame_ids = []
        view_labels = []
        yaw_degs = []

        for batch in data_loader:
            batch_count = batch['sat'].shape[0]
            remaining = self.num_visualizations - len(frame_ids)
            if remaining <= 0:
                break

            take = min(remaining, batch_count)
            sat_chunks.append(batch['sat'][:take])
            target_chunks.append(batch['image'][:take])

            front_bev_xy = batch.get('front_bev_xy')
            if front_bev_xy is not None:
                front_bev_xy_chunks.append(front_bev_xy[:take])
            front_ground_valid_mask = batch.get('front_ground_valid_mask')
            if front_ground_valid_mask is not None:
                front_ground_valid_mask_chunks.append(front_ground_valid_mask[:take])

            for cam_key, chunks in [
                ('K', K_chunks),
                ('T_cam_to_world', T_cam_to_world_chunks),
                ('T_imu_to_world', T_imu_to_world_chunks),
            ]:
                cam_val = batch.get(cam_key)
                if cam_val is not None:
                    chunks.append(cam_val[:take])
            camera_height = batch.get('camera_height_m')
            if torch.is_tensor(camera_height):
                camera_height_chunks.append(camera_height[:take])

            batch_frame_ids = batch.get('frame_id')
            if batch_frame_ids is None:
                frame_ids.extend([None] * take)
            else:
                frame_ids.extend(list(batch_frame_ids[:take]))
            view_labels.extend([None] * take)
            yaw_degs.extend([None] * take)

            if len(frame_ids) >= self.num_visualizations:
                break

        if not sat_chunks:
            return None

        result = {
            "sat": torch.cat(sat_chunks, dim=0),
            "image": torch.cat(target_chunks, dim=0),
            "front_bev_xy": torch.cat(front_bev_xy_chunks, dim=0) if front_bev_xy_chunks else None,
            "front_ground_valid_mask": (
                torch.cat(front_ground_valid_mask_chunks, dim=0)
                if front_ground_valid_mask_chunks else None
            ),
            "frame_ids": frame_ids,
            "view_labels": view_labels,
            "yaw_degs": yaw_degs,
        }
        if camera_height_chunks:
            result["camera_height_m"] = torch.cat(camera_height_chunks, dim=0)
        for cam_key, chunks in [
            ('K', K_chunks),
            ('T_cam_to_world', T_cam_to_world_chunks),
            ('T_imu_to_world', T_imu_to_world_chunks),
        ]:
            if chunks:
                result[cam_key] = torch.cat(chunks, dim=0)
        return result

    @torch.no_grad()
    def _save_visualizations(self, epoch: int):
        data_loader = self.val_dataloader if self.val_dataloader is not None else self.train_dataloader
        if data_loader is None:
            return

        fixed_items = self._collect_fixed_yaw_visualization_items(data_loader)
        visualization_batch = (
            self._stack_visualization_items(fixed_items)
            if fixed_items is not None
            else None
        )
        if visualization_batch is None:
            visualization_batch = self._collect_fallback_visualization_batch(data_loader)
        if visualization_batch is None:
            return

        sat_images = visualization_batch["sat"].to(self.device)
        target_images = visualization_batch["image"].to(self.device)
        front_bev_xy = visualization_batch["front_bev_xy"]
        if front_bev_xy is not None:
            front_bev_xy = front_bev_xy.to(self.device)
        front_ground_valid_mask = visualization_batch["front_ground_valid_mask"]
        front_ground_valid_mask = (
            front_ground_valid_mask.to(self.device) if front_ground_valid_mask is not None else None
        )
        frame_ids = visualization_batch["frame_ids"]
        view_labels = visualization_batch["view_labels"]
        yaw_degs = visualization_batch["yaw_degs"]

        K = visualization_batch.get("K")
        if K is not None:
            K = K.to(self.device)
        T_cam_to_world = visualization_batch.get("T_cam_to_world")
        if T_cam_to_world is not None:
            T_cam_to_world = T_cam_to_world.to(self.device)
        T_imu_to_world = visualization_batch.get("T_imu_to_world")
        if T_imu_to_world is not None:
            T_imu_to_world = T_imu_to_world.to(self.device)
        camera_height_m = visualization_batch.get("camera_height_m")
        if torch.is_tensor(camera_height_m):
            camera_height_m = camera_height_m.to(self.device)
        target_size = (int(target_images.shape[2]), int(target_images.shape[3]))

        generator_device = self.device if self.device.startswith("cuda") else "cpu"

        eval_model = self.unwrapped_model
        was_training = eval_model.training
        eval_model.eval()
        sat_state = eval_model.encode_satellite(
            sat_images,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            image_size=target_size,
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.visualization_seed)
        generated_chunks = []
        attention_debug_by_sample: List[Dict[str, Any]] = []
        unet = getattr(eval_model, "unet", None)
        for idx in range(sat_images.shape[0]):
            view_sat_state = SatelliteMemoryState(
                tokens=sat_state.tokens[idx:idx + 1],
                xy=sat_state.xy[idx:idx + 1],
                bev_coords=(
                    sat_state.bev_coords[idx:idx + 1]
                    if sat_state.bev_coords is not None
                    else None
                ),
                perspective_uv=(
                    sat_state.perspective_uv[idx:idx + 1]
                    if sat_state.perspective_uv is not None
                    else None
                ),
                perspective_valid=(
                    sat_state.perspective_valid[idx:idx + 1]
                    if sat_state.perspective_valid is not None
                    else None
                ),
            )
            attention_debug: Dict[str, Any] = {}
            if unet is not None and hasattr(unet, "enable_attention_debug"):
                unet.enable_attention_debug(
                    layers=self.attention_visualization_layers,
                    storage=attention_debug,
                )
            try:
                generated_view, _ = eval_model.generate_with_satellite_state(
                    view_sat_state,
                    target_size=tuple(target_images.shape[-2:]),
                    num_inference_steps=self.visualization_inference_steps,
                    guidance_scale=self.visualization_guidance_scale,
                    generator=generator,
                )
            finally:
                if unet is not None and hasattr(unet, "disable_attention_debug"):
                    unet.disable_attention_debug()
            generated_chunks.append(generated_view)
            attention_debug_by_sample.append(attention_debug)
        generated_images = torch.cat(generated_chunks, dim=0)
        if was_training:
            eval_model.train()

        epoch_dir = self.visualization_dir / f"epoch_{epoch + 1:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        attention_dir = epoch_dir / "attention"
        comparison_images = []
        captions = []

        for idx in range(sat_images.shape[0]):
            frame_id = frame_ids[idx]
            frame_suffix = f"_frame_{int(frame_id):010d}" if frame_id is not None else ""
            view_label = view_labels[idx]
            view_suffix = f"_{view_label}" if view_label else ""
            comparison = self._compose_visualization(
                sat_images[idx],
                generated_images[idx],
                target_images[idx],
                front_bev_xy=front_bev_xy[idx] if front_bev_xy is not None else None,
                front_ground_valid_mask=(
                    front_ground_valid_mask[idx] if front_ground_valid_mask is not None else None
                ),
                view_label=view_label,
                yaw_deg=yaw_degs[idx],
            )
            comparison.save(epoch_dir / f"sample_{idx:02d}{view_suffix}{frame_suffix}.png")
            attention_prefix = f"sample_{idx:02d}{view_suffix}{frame_suffix}"
            self._save_attention_debug_visualizations(
                attention_debug=attention_debug_by_sample[idx],
                sat_image=sat_images[idx],
                front_bev_xy=front_bev_xy[idx] if front_bev_xy is not None else None,
                front_ground_valid_mask=(
                    front_ground_valid_mask[idx] if front_ground_valid_mask is not None else None
                ),
                output_dir=attention_dir,
                prefix=attention_prefix,
            )
            comparison_images.append(comparison)
            caption = f"epoch={epoch + 1} sample={idx:02d}"
            if view_label:
                caption += f" view={view_label}"
            if yaw_degs[idx] is not None:
                caption += f" yaw={float(yaw_degs[idx]):g}"
            if frame_id is not None:
                caption += f" frame={int(frame_id):010d}"
            captions.append(caption)

        logger.info(f"Saved visualizations: {epoch_dir}")
        self._log_visualizations(
            comparison_images,
            captions,
            step=self._global_step(epoch, max(0, len(self.train_dataloader) - 1)),
        )

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load from checkpoint and return the next zero-based epoch index."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        load_model_state_dict(self.unwrapped_model, checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        checkpoint_epoch = checkpoint.get('epoch')
        start_epoch = int(checkpoint_epoch) + 1 if checkpoint_epoch is not None else 0
        if checkpoint_epoch is None:
            logger.info(
                "Checkpoint loaded: %s (no epoch metadata found; restarting epoch count from 1)",
                checkpoint_path,
            )
        else:
            logger.info(
                "Checkpoint loaded: %s (resuming at epoch %d/%d)",
                checkpoint_path,
                start_epoch + 1,
                self.num_train_epochs,
            )
        return start_epoch
