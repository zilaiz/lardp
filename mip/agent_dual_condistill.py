"""Torch training agent for dual-network condistill behavior cloning.

Teacher and student are fully separate trainable networks.
Teacher sees obs + extra_cond; student sees only obs.
No dropout annealing needed.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import loguru
import torch
import torch.nn as nn
from tensordict import TensorDict

from mip.config import Config
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_loss_fn
from mip.network_utils import (
    get_encoder,
    get_extra_cond_encoder,
    get_network,
)
from mip.samplers import get_sampler
from mip.torch_utils import report_parameters


class TrainingAgentDualCondistill:
    """Training agent for dual-network condistill with flow matching."""

    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.loss_fn = get_loss_fn(config.optimization.loss_type)
        self.sampler = get_sampler(config.optimization.loss_type)
        self.interpolant = Interpolant(config.optimization.interp_type)

        # Teacher networks (trainable)
        net_teacher = get_network(config.network, config.task)
        report_parameters(net_teacher, model_name="Teacher Action Network")
        self.flow_map_teacher = FlowMap(net_teacher).to(config.optimization.device)
        self.encoder_teacher = get_encoder(config.network, config.task).to(
            config.optimization.device
        )
        self.extra_cond_encoder = get_extra_cond_encoder(config.network, config.task).to(
            config.optimization.device
        )
        report_parameters(self.encoder_teacher, model_name="Teacher Encoder Network")
        report_parameters(self.extra_cond_encoder, model_name="Extra Cond Encoder Network")

        # Student networks (trainable)
        net_student = get_network(config.network, config.task)
        report_parameters(net_student, model_name="Student Action Network")
        self.flow_map_student = FlowMap(net_student).to(config.optimization.device)
        self.encoder_student = get_encoder(config.network, config.task).to(
            config.optimization.device
        )
        report_parameters(self.encoder_student, model_name="Student Encoder Network")

        # Student EMA (for stable inference)
        self.encoder_student_ema = deepcopy(self.encoder_student).requires_grad_(False)
        self.flow_map_student_ema = deepcopy(self.flow_map_student).requires_grad_(False)

        # Create detached models for CUDA graphs (student only, used at inference)
        self.use_cudagraphs = config.optimization.use_cudagraphs
        if self.use_cudagraphs:
            self.flow_map_student_detach = deepcopy(self.flow_map_student).requires_grad_(False)
            self.encoder_student_detach = deepcopy(self.encoder_student).requires_grad_(False)
            self.flow_map_student_ema_detach = deepcopy(self.flow_map_student_ema).requires_grad_(False)
            self.encoder_student_ema_detach = deepcopy(self.encoder_student_ema).requires_grad_(False)
        else:
            self.flow_map_student_detach = None
            self.encoder_student_detach = None
            self.flow_map_student_ema_detach = None
            self.encoder_student_ema_detach = None

        # Single optimizer over all trainable parameters
        params = (
            list(self.encoder_teacher.parameters())
            + list(self.extra_cond_encoder.parameters())
            + list(self.flow_map_teacher.parameters())
            + list(self.encoder_student.parameters())
            + list(self.flow_map_student.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

        # Store obs keys if using image observations (for CUDA graph compatibility)
        if hasattr(config.task, "shape_meta") and "obs" in config.task.shape_meta:
            self.obs_keys = sorted(config.task.shape_meta["obs"].keys())
        else:
            self.obs_keys = None

        # Compile training and sampling functions for faster execution
        self.use_compile = config.optimization.use_compile
        self.compile_mode = config.optimization.compile_mode
        self.__compile__()

    def __compile__(self):
        """Compile training and inference functions."""
        loguru.logger.info(
            f"Compile: {self.use_compile} | "
            f"Compile mode: {self.compile_mode} | "
            f"CUDA graphs: {self.use_cudagraphs}"
        )

        self._update_impl = self._create_update_impl()
        self._sample_fn = self._sample_impl

        # Step 1: Setup CUDA graphs - copy params to detached models
        if self.use_cudagraphs:
            from tensordict import from_module

            loguru.logger.info(
                "Setting up CUDA graphs - copying parameters to detached models"
            )
            from_module(self.flow_map_student).data.to_module(self.flow_map_student_detach)
            from_module(self.encoder_student).data.to_module(self.encoder_student_detach)
            from_module(self.flow_map_student_ema).data.to_module(self.flow_map_student_ema_detach)
            from_module(self.encoder_student_ema).data.to_module(self.encoder_student_ema_detach)

            self._sample_fn = self._inference_mode()(self._sample_impl)

        # Step 2: Compile with torch.compile
        if self.use_compile:
            loguru.logger.info(
                "Compiling entire update loop (forward+backward+optimizer) with torch.compile"
            )
            self._compiled_update = torch.compile(
                self._update_impl, mode=self.compile_mode
            )
            loguru.logger.info("Compiling sampler with torch.compile")
            self._compiled_sampler = torch.compile(
                self._sample_fn, mode=self.compile_mode
            )
            loguru.logger.info("Successfully compiled models")
        else:
            self._compiled_update = self._update_impl
            self._compiled_sampler = self._sample_fn

        # Step 3: Wrap with CudaGraphModule for CUDA graph capture
        if self.use_cudagraphs:
            from tensordict.nn import CudaGraphModule

            loguru.logger.info(
                "Wrapping update function with CudaGraphModule for CUDA graph capture"
            )
            self._compiled_update = CudaGraphModule(
                self._compiled_update, in_keys=[], out_keys=[]
            )
            loguru.logger.info("CUDA graph setup complete")

    @contextmanager
    def _inference_mode(self):
        """Context manager to switch to inference models (student only)."""
        if self.use_cudagraphs:
            flow_map_student_backup = self.flow_map_student
            encoder_student_backup = self.encoder_student
            flow_map_student_ema_backup = self.flow_map_student_ema
            encoder_student_ema_backup = self.encoder_student_ema

            self.flow_map_student = self.flow_map_student_detach
            self.encoder_student = self.encoder_student_detach
            self.flow_map_student_ema = self.flow_map_student_ema_detach
            self.encoder_student_ema = self.encoder_student_ema_detach

            try:
                yield
            finally:
                self.flow_map_student = flow_map_student_backup
                self.encoder_student = encoder_student_backup
                self.flow_map_student_ema = flow_map_student_ema_backup
                self.encoder_student_ema = encoder_student_ema_backup
        else:
            was_training = self.flow_map_student.training
            try:
                self.flow_map_student.eval()
                self.encoder_student.eval()
                self.flow_map_student_ema.eval()
                self.encoder_student_ema.eval()
                yield
            finally:
                if was_training:
                    self.flow_map_student.train()
                    self.encoder_student.train()

    def _sync_detached_models(self):
        """Synchronize detached models with main models for CUDA graphs."""
        if self.use_cudagraphs:
            from tensordict import from_module

            from_module(self.flow_map_student).data.to_module(self.flow_map_student_detach)
            from_module(self.encoder_student).data.to_module(self.encoder_student_detach)
            from_module(self.flow_map_student_ema).data.to_module(self.flow_map_student_ema_detach)
            from_module(self.encoder_student_ema).data.to_module(self.encoder_student_ema_detach)

    def _create_update_impl(self):
        """Create the update implementation function that will be compiled."""

        def update_impl(data: TensorDict):
            act = data["act"]
            obs = data["obs"]
            extra_cond = data["extra_cond"]
            delta_t = data["delta_t"]

            # Forward pass: dual condistill loss
            teacher_dp_loss, student_dp_loss, projection_loss, _info = self.loss_fn(
                self.config.optimization,
                self.flow_map_teacher,
                self.encoder_teacher,
                self.extra_cond_encoder,
                self.flow_map_student,
                self.encoder_student,
                self.interpolant,
                act,
                obs,
                extra_cond,
                delta_t,
            )

            student_loss = student_dp_loss + projection_loss
            loss = teacher_dp_loss + student_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            params = (
                list(self.encoder_teacher.parameters())
                + list(self.extra_cond_encoder.parameters())
                + list(self.flow_map_teacher.parameters())
                + list(self.encoder_student.parameters())
                + list(self.flow_map_student.parameters())
            )
            if self.config.optimization.grad_clip_norm:
                grad_norm = nn.utils.clip_grad_norm_(
                    params, self.config.optimization.grad_clip_norm
                )
            else:
                grad_norm = torch.tensor(0.0, device=loss.device)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # EMA update (student only)
            if self.config.optimization.ema_rate < 1:
                self._ema_update_impl()

            result_dict = {
                "teacher_dp_loss": teacher_dp_loss.detach(),
                "student_dp_loss": student_dp_loss.detach(),
                "student_loss": student_loss.detach(),
                "repa_loss": projection_loss.detach(),
                "grad_norm": grad_norm.detach(),
            }
            for k, v in _info.items():
                result_dict[k] = torch.tensor(v, device=loss.device)
            result = TensorDict(result_dict, batch_size=())
            return result

        return update_impl

    def _ema_update_impl(self):
        """EMA update for student models only."""
        params = list(self.encoder_student.parameters()) + list(self.flow_map_student.parameters())
        params_ema = list(self.encoder_student_ema.parameters()) + list(
            self.flow_map_student_ema.parameters()
        )
        with torch.no_grad():
            for p, p_ema in zip(params, params_ema, strict=False):
                p_ema.data.mul_(self.config.optimization.ema_rate).add_(
                    p.data, alpha=1.0 - self.config.optimization.ema_rate
                )

    def update(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict | TensorDict,
        extra_cond: torch.Tensor | dict | TensorDict,
        delta_t: torch.Tensor,
    ):
        """Update model parameters with a training batch.

        Args:
            act: Action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor or dict of tensors
            extra_cond: Extra condition tensor or dict of tensors
            delta_t: Time step differences of shape (batch_size,)

        Returns:
            Dictionary containing loss and gradient norm statistics
        """
        # Check batch size consistency for CUDA graphs
        if self.use_cudagraphs:
            if not hasattr(self, "_expected_batch_size"):
                self._expected_batch_size = act.shape[0]
            elif act.shape[0] != self._expected_batch_size:
                raise ValueError(
                    f"CUDA graphs require static batch sizes. "
                    f"Expected {self._expected_batch_size}, got {act.shape[0]}. "
                    f"Make sure your dataloader has drop_last=True."
                )

        # Mark CUDA graph step boundary if using compile
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        data = TensorDict(
            {
                "act": act,
                "obs": obs,
                "extra_cond": extra_cond,
                "delta_t": delta_t,
            },
            batch_size=act.shape[0],
        )
        result = self._compiled_update(data)

        out = {
            "teacher_dp_loss": result["teacher_dp_loss"],
            "dp_loss": result["student_dp_loss"],
            "repa_loss": result["repa_loss"],
            "loss": result["student_loss"],
            "grad_norm": result["grad_norm"],
        }
        for k in result.keys():  # noqa: SIM118
            if k.startswith('diag/'):
                out[k] = result[k]
        return out

    def _sample_impl(
        self,
        config,
        flow_map,
        encoder,
        act_0: torch.Tensor,
        obs: torch.Tensor,
    ):
        """Internal sampling implementation. Student only, no padding."""
        return self.sampler(config, flow_map, encoder, act_0, obs, padding_len=0)

    def sample(
        self,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        num_steps: int = -1,
        use_ema: bool = True,
    ):
        """Sample actions using student network only.

        Args:
            act_0: Initial action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor of shape (batch_size, To, obs_dim)
            num_steps: Number of sampling steps (default: use config value)
            use_ema: Whether to use EMA parameters for sampling

        Returns:
            Sampled action tensor of shape (batch_size, Ta, act_dim)
        """
        # Sync detached models if using CUDA graphs before inference
        if self.use_cudagraphs:
            self._sync_detached_models()

        if num_steps >= 1:
            config = deepcopy(self.config.optimization)
            config.num_steps = int(num_steps)
        else:
            config = self.config.optimization

        # Use student (or student EMA) for inference
        if self.config.optimization.ema_rate < 1 and use_ema:
            flow_map = self.flow_map_student_ema
            encoder = self.encoder_student_ema
        else:
            flow_map = self.flow_map_student
            encoder = self.encoder_student

        with torch.no_grad():
            if not self.use_cudagraphs:
                with self._inference_mode():
                    act = self._compiled_sampler(config, flow_map, encoder, act_0, obs)
            else:
                act = self._compiled_sampler(config, flow_map, encoder, act_0, obs)
        return act

    def save(self, path: str, training_state: dict = None):
        """Save agent models to path."""
        checkpoint = {
            "flow_map_teacher": self.flow_map_teacher.state_dict(),
            "encoder_teacher": self.encoder_teacher.state_dict(),
            "extra_cond_encoder": self.extra_cond_encoder.state_dict(),
            "flow_map_student": self.flow_map_student.state_dict(),
            "encoder_student": self.encoder_student.state_dict(),
            "encoder_student_ema": self.encoder_student_ema.state_dict(),
            "flow_map_student_ema": self.flow_map_student_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if training_state is not None:
            checkpoint["training_state"] = training_state
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False):
        """Load agent models from path."""
        state_dict = torch.load(
            path, map_location=self.config.optimization.device, weights_only=False
        )
        self.flow_map_teacher.load_state_dict(state_dict["flow_map_teacher"])
        self.encoder_teacher.load_state_dict(state_dict["encoder_teacher"])
        self.extra_cond_encoder.load_state_dict(state_dict["extra_cond_encoder"])
        self.flow_map_student.load_state_dict(state_dict["flow_map_student"])
        self.encoder_student.load_state_dict(state_dict["encoder_student"])
        self.encoder_student_ema.load_state_dict(state_dict["encoder_student_ema"])
        self.flow_map_student_ema.load_state_dict(state_dict["flow_map_student_ema"])

        if load_optimizer and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            loguru.logger.info("Loaded optimizer state")

        # Recompile after loading
        self.__compile__()

        training_state = state_dict.get("training_state", None)
        if training_state:
            loguru.logger.info(
                f"Loaded training state from step {training_state.get('n_gradient_step', 'unknown')}"
            )
        return training_state

    def eval(self):
        """Set all models to evaluation mode."""
        self.flow_map_teacher.eval()
        self.encoder_teacher.eval()
        self.extra_cond_encoder.eval()
        self.flow_map_student.eval()
        self.encoder_student.eval()
        self.flow_map_student_ema.eval()
        self.encoder_student_ema.eval()

    def train(self):
        """Set all models to training mode."""
        self.flow_map_teacher.train()
        self.encoder_teacher.train()
        self.extra_cond_encoder.train()
        self.flow_map_student.train()
        self.encoder_student.train()
        self.flow_map_student_ema.train()
        self.encoder_student_ema.train()
