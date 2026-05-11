"""Losses for iterative policy training."""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from mip.config import OptimizationConfig
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant


def get_norm(x: torch.Tensor, norm_type: str) -> torch.Tensor:
    if norm_type == "l2":
        # squared L2 (no sqrt)
        return torch.sum(x * x, dim=-1)
    elif norm_type == "l1":
        return torch.sum(torch.abs(x), dim=-1)
    elif norm_type == "smooth_l1":
        # per-element smooth L1, then sum over last dim
        return torch.sum(
            F.smooth_l1_loss(x, torch.zeros_like(x), reduction="none"), dim=-1
        )
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented.")


def get_loss_fn(loss_type: str) -> Callable:
    if loss_type == "flow":
        return flow_loss
    elif loss_type == "flow_repa":
        return flow_repa_loss
    elif loss_type == "flow_condistill":
        return flow_condistill_loss
    elif loss_type == "flow_fast_condistill":
        return flow_fast_condistill_loss
    elif loss_type == "flow_dual_condistill":
        return flow_dual_condistill_loss
    elif loss_type == "flow_reg":
        return flow_reg_loss
    elif loss_type == "flow_beta":
        return flow_beta_loss
    elif loss_type == "flow_ns":
        return flow_ns_loss
    elif loss_type == "flow_beta_ll":
        return flow_beta_ll_loss
    elif loss_type == "regression":
        return regression_loss
    elif loss_type == "straight_flow":
        return straight_flow_loss
    elif loss_type == "tsd":
        return tsd_loss
    elif loss_type == "mip":
        return mip_loss
    elif loss_type == "lmd":
        return lmd_loss
    elif loss_type == "ctm":
        return ctm_loss
    elif loss_type == "psd":
        return psd_loss
    elif loss_type == "lsd":
        return lsd_loss
    elif loss_type == "esd":
        return esd_loss
    elif loss_type == "mf":
        return mf_loss
    elif loss_type == "goal_predictor":
        return flow_loss  # actual loss computed inline in GoalPredictorAgent
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def repa_loss(zs_tilde, tgt_act_reps):
    proj_loss = 0.
    for z, z_tilde in zip(tgt_act_reps, zs_tilde, strict=False):
        assert z.shape == z_tilde.shape
        z = F.normalize(z, dim=-1)
        z_tilde = F.normalize(z_tilde, dim=-1)
        proj_loss += -(z * z_tilde).sum(dim=-1).mean()
    proj_loss /= len(tgt_act_reps)
    return proj_loss


def flow_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Flow model loss, matching the velocity field.

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.

    Returns:
        float: the loss
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def flow_beta_ll_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs,
    delta_t: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Flow-Beta loss plus local-linearity regularizer for IDM training.

    When `config.local_linearity_coef > 0` the training script builds obs per
    image key as `[To_0, ..., To_{n-1}, o_k, goal]` (shape `(B, To+2, ...)`).
    We run one forward through the raw inner encoder on all `To+2` frames,
    then:

    - For action decoding, we drop the intermediate slot and feed the
      `(B, To+1, D)` stack `[To_0, ..., To_{n-1}, goal]` to the flow map —
      identical conditioning to standard `flow_beta_loss`.
    - For the regularizer, we reuse the last three embedding slots
      `[To_{n-1}, o_k, goal]` and push the two segment velocities
      `v1 = enc(o_k) - enc(To_{n-1})`, `v2 = enc(goal) - enc(o_k)` toward
      alignment by minimizing `1 - cos(v1, v2)`.

    Goal dropout from a wrapping `GoalDropoutEncoder` is applied manually
    here (the main forward bypasses that wrap since it can't accept
    `To + 2` frames). When `coef == 0` we fall back to `flow_beta_loss`.
    """
    coef = config.local_linearity_coef
    if coef <= 0:
        return flow_beta_loss(
            config, flow_map, encoder, interp, act, obs, delta_t
        )

    # t = 0.999 * (1 - u), u ~ Beta(1.5, 1.0); biases toward t=0 (noise).
    u = torch.distributions.Beta(1.5, 1.0).sample(delta_t.shape).to(delta_t.device)
    t = 0.999 * (1.0 - u)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # raw encoder forward on all To+2 frames (bypass GoalDropoutEncoder wrap)
    raw_encoder = encoder.encoder if hasattr(encoder, "encoder") else encoder
    full_emb = raw_encoder(obs, None)  # (B, To+2, D)

    # action-decoding branch: drop the intermediate slot
    obs_emb = torch.cat([full_emb[:, :-2], full_emb[:, -1:]], dim=1)  # (B, To+1, D)

    # delegate goal dropout to GoalDropoutEncoder (single source of truth);
    # no-op when encoder isn't a GoalDropoutEncoder or dropout is disabled.
    if hasattr(encoder, "apply_goal_dropout"):
        obs_emb = encoder.apply_goal_dropout(obs_emb)

    # flow prediction
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t = flow_map.get_velocity(t, act_t, obs_emb)
    flow_l = config.loss_scale * torch.mean(
        get_norm(b_t - act_t_dot, config.norm_type)
    )

    # local-linearity branch: last three slots are [To_{n-1}, o_k, goal]
    to1_emb = full_emb[:, -3]
    inter_emb = full_emb[:, -2]
    goal_emb = full_emb[:, -1]
    v1 = inter_emb - to1_emb
    v2 = goal_emb - inter_emb
    ll_l = coef * (1.0 - F.cosine_similarity(v1, v2, dim=-1)).mean()

    total = flow_l + ll_l
    info = {
        "flow_loss": flow_l.detach(),
        "local_linearity_loss": ll_l.detach(),
    }
    return total, info


def _ns_apply_t_shift(t: torch.Tensor, alpha: float) -> torch.Tensor:
    """SD3-style time shift t' = a*t / (1 + (a-1)*t). Identity at a=1."""
    if alpha == 1.0:
        return t
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def _ns_sample_base_t(
    delta_t: torch.Tensor,
    lo: float,
    hi: float,
    t_dist: str,
    mu: float,
    sigma: float,
) -> torch.Tensor:
    """Sample base flow time before per-stream shift is applied.

    Uses ``empty + uniform_/normal_`` so the call is CUDA-graph compatible
    (matches the pattern in ``flow_loss`` / ``flow_beta_loss``).
    """
    if t_dist == "uniform":
        return torch.empty_like(delta_t).uniform_(lo, hi)
    if t_dist == "logit_normal":
        z = torch.empty_like(delta_t).normal_(mu, sigma)
        return torch.sigmoid(z).clamp(lo, hi)
    raise ValueError(
        f"policy_t_dist must be 'uniform' or 'logit_normal'; got {t_dist!r}"
    )


def flow_ns_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Flow matching loss with SD3-style noise-shift on the flow time.

    Identical to ``flow_loss`` except ``t`` is drawn from a configurable
    base distribution (``policy_t_dist``: "uniform" | "logit_normal"),
    clamped to ``[policy_t_eps, 1 - policy_t_eps]``, then warped through
    the SD3 shift ``t' = a*t / (1 + (a-1)*t)`` with ``a = policy_t_shift``.
    Identity defaults (``shift=1.0``, ``dist=uniform``, ``eps=0.0``) recover
    plain ``flow_loss`` exactly.

    Pairs with ``ode_ns_sampler`` so inference walks the same shift-warped
    grid the trunk was trained on.
    """
    eps = float(config.policy_t_eps)
    lo, hi = eps, 1.0 - eps
    t_base = _ns_sample_base_t(
        delta_t,
        lo, hi,
        config.policy_t_dist,
        float(config.policy_t_dist_mu),
        float(config.policy_t_dist_sigma),
    )
    t = _ns_apply_t_shift(t_base, float(config.policy_t_shift))

    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    obs_emb = encoder(obs, None)

    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t = flow_map.get_velocity(t, act_t, obs_emb)

    loss = get_norm(b_t - act_t_dot, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {"t_flow_mean": t.mean().detach()}


def flow_beta_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Flow model loss with Beta distribution timestep sampling.

    Samples t = 0.999 * (1 - u),  u ~ Beta(1.5, 1.0). Biases training toward
    small t = noise end (mip's interpolant is (1-t)*noise + t*data, so t=0
    is pure noise). The 0.999 cap keeps a sliver of noise mixed in at the
    data end for numerical stability. Matches PI-0 / multitask_dit_policy.

    Args:
        config: optimization config
        flow_map (FlowMap): the flow map
        encoder (BaseEncoder): the encoder
        interp (Interpolant): the interpolant
        act (torch.Tensor): the target action
        obs (torch.Tensor): the observation
        delta_t (torch.Tensor): the time step difference

    Returns:
        float: the loss
    """
    # u ~ Beta(1.5, 1.0) is biased toward 1; 0.999*(1-u) puts mass near t=0
    # (noise end under mip's (1-t)*noise + t*data interpolant).
    u = torch.distributions.Beta(1.5, 1.0).sample(delta_t.shape).to(delta_t.device)
    t = 0.999 * (1.0 - u)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def flow_repa_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
    tgt_act_reps: torch.Tensor,
) -> float:
    """Flow model loss, matching the velocity field.

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.
        tgt_act_reps (torch.Tensor): target action representations

    Returns:
        float: the loss
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t, zs_tilde = flow_map.get_velocity_repa(t, act_t, obs_emb, align_depth=config.s_align_depth)

    if zs_tilde is None:
        raise ValueError(
            "zs_tilde is None — set network.align_depth to a value in [1, depth] to enable REPA"
        )

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)

    projection_loss = repa_loss(zs_tilde, tgt_act_reps)
    projection_loss *= config.repa_scale
    return loss, projection_loss, {}


def flow_condistill_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    extra_cond_encoder: BaseEncoder,
    flow_map_ema: FlowMap,
    encoder_ema: BaseEncoder,
    extra_cond_encoder_ema: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    extra_cond: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Flow model loss, matching the velocity field.

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.
        tgt_act_reps (torch.Tensor): target action representations

    Returns:
        float: the loss
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)  # encoder_dropout: 0
    extra_cond_emb = extra_cond_encoder(extra_cond, None)  # extra_cond_encoder_dropout: annealed from 0
    dummy_cond_emb = torch.zeros_like(extra_cond_emb)

    full_obs_emb = torch.cat([obs_emb, extra_cond_emb], dim=1)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t, _ = flow_map.get_velocity_repa(t, act_t, full_obs_emb, align_depth=None)

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)

    # condition-guided distillation
    _, zs_tilde_student = flow_map.get_velocity_repa(t, act_t, torch.cat([obs_emb, dummy_cond_emb], dim=1), align_depth=config.s_align_depth)

    with torch.no_grad():
        obs_emb_ema = encoder_ema(obs, None)  # encoder dropout is already 0
        extra_cond_encoder_ema.eval()
        extra_cond_emb_ema = extra_cond_encoder_ema(extra_cond, None)  # make sure input masks have no dropout
        extra_cond_encoder_ema.train()
        full_obs_emb_ema = torch.cat([obs_emb_ema, extra_cond_emb_ema], dim=1)
        flow_map_ema.eval()
        _, zs_tilde_teacher = flow_map_ema.get_velocity_repa(t, act_t, full_obs_emb_ema, align_depth=config.t_align_depth)
        flow_map_ema.train()

    if zs_tilde_student is None or zs_tilde_teacher is None:
        raise ValueError(
            "zs_tilde is None — set network.align_depth to a value in [1, depth] to enable representation extraction"
        )
    projection_loss = repa_loss(zs_tilde_student, zs_tilde_teacher)
    projection_loss *= config.repa_scale
    return loss, projection_loss, {}


def flow_fast_condistill_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    extra_cond_encoder: BaseEncoder,
    flow_map_ema: FlowMap,
    encoder_ema: BaseEncoder,
    extra_cond_encoder_ema: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    extra_cond: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Flow model loss, matching the velocity field.

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.
        tgt_act_reps (torch.Tensor): target action representations

    Returns:
        float: the loss
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)  # encoder_dropout: 0
    extra_cond_emb = extra_cond_encoder(extra_cond, None)  # extra_cond_encoder_dropout: annealed from 0

    full_obs_emb = torch.cat([obs_emb, extra_cond_emb], dim=1)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t, zs_tilde_student = flow_map.get_velocity_repa(t, act_t, full_obs_emb, align_depth=config.s_align_depth)

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)

    # condition-guided distillation
    with torch.no_grad():
        obs_emb_ema = encoder_ema(obs, None)  # encoder dropout is already 0
        extra_cond_encoder_ema.eval()
        extra_cond_emb_ema = extra_cond_encoder_ema(extra_cond, None)  # make sure input masks have no dropout
        extra_cond_encoder_ema.train()
        full_obs_emb_ema = torch.cat([obs_emb_ema, extra_cond_emb_ema], dim=1)
        flow_map_ema.eval()
        _, zs_tilde_teacher = flow_map_ema.get_velocity_repa(t, act_t, full_obs_emb_ema, align_depth=config.t_align_depth)
        flow_map_ema.train()

    if zs_tilde_student is None or zs_tilde_teacher is None:
        raise ValueError(
            "zs_tilde is None — set network.align_depth to a value in [1, depth] to enable representation extraction"
        )
    projection_loss = repa_loss(zs_tilde_student, zs_tilde_teacher)
    projection_loss *= config.repa_scale
    return loss, projection_loss, {}


def flow_dual_condistill_loss(
    config: OptimizationConfig,
    flow_map_teacher: FlowMap,
    encoder_teacher: BaseEncoder,
    extra_cond_encoder: BaseEncoder,
    flow_map_student: FlowMap,
    encoder_student: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    extra_cond: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Dual-network condistill loss.

    Teacher (trainable) sees full condition (obs + extra_cond).
    Student (trainable) sees only obs (no extra_cond, no zero padding).
    REPA aligns student reps to detached teacher reps.
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)

    # Always extract at all decoder depths; only optimize target depths via REPA.
    n_decoder_layers = flow_map_teacher.net.n_layers
    all_depths = list(range(1, n_decoder_layers + 1))

    # Normalize target depths to index sets for filtering
    s_depths = [config.s_align_depth] if isinstance(config.s_align_depth, int) else list(config.s_align_depth)
    t_depths = [config.t_align_depth] if isinstance(config.t_align_depth, int) else list(config.t_align_depth)

    # Teacher forward (eval mode, all depths)
    flow_map_teacher.eval()
    obs_emb_t = encoder_teacher(obs, None)
    extra_cond_emb = extra_cond_encoder(extra_cond, None)
    full_obs_emb = torch.cat([obs_emb_t, extra_cond_emb], dim=1)
    b_t_teacher, zs_teacher_all = flow_map_teacher.get_velocity_repa(
        t, act_t, full_obs_emb, align_depth=all_depths
    )
    flow_map_teacher.train()

    teacher_loss = config.loss_scale * torch.mean(
        get_norm(b_t_teacher - act_t_dot, config.norm_type)
    )

    # Student forward (train mode, all depths)
    obs_emb_s = encoder_student(obs, None)
    b_t_student, zs_student_all = flow_map_student.get_velocity_repa(
        t, act_t, obs_emb_s, align_depth=all_depths
    )

    student_loss = config.loss_scale * torch.mean(
        get_norm(b_t_student - act_t_dot, config.norm_type)
    )

    # REPA loss: only on target depths
    # depth i is at index i-1 in the all_depths list
    zs_student_target = [zs_student_all[d - 1] for d in s_depths]
    zs_teacher_target = [zs_teacher_all[d - 1] for d in t_depths]

    if zs_student_target is None or zs_teacher_target is None:
        raise ValueError(
            "zs_tilde is None — set network.align_depth to a value in [1, depth] to enable representation extraction"
        )
    projection_loss = repa_loss(zs_student_target, [z.detach() for z in zs_teacher_target])
    projection_loss *= config.repa_scale

    info = {}

    # Diagnostic: per-depth REPA loss with matched vs shuffled extra_cond.
    # Reuses already-extracted teacher/student all-depth reps; only extra cost
    # is one shuffled teacher forward pass.
    # repa_gap = repa_shuffled - repa_matched tells us how much signal each depth has.
    if config.diagnose_teacher_delta:
        with torch.no_grad():
            perm = torch.randperm(extra_cond_emb.shape[0], device=extra_cond_emb.device)
            shuffled_cond_emb = extra_cond_emb[perm]
            shuffled_obs_emb = torch.cat([obs_emb_t, shuffled_cond_emb], dim=1)
            flow_map_teacher.eval()
            _, zs_shuf = flow_map_teacher.get_velocity_repa(
                t, act_t, shuffled_obs_emb, align_depth=all_depths
            )
            flow_map_teacher.train()
            # Stack all: (n_depths, B, Ta, d_model)
            zs_s = torch.stack([z.detach() for z in zs_student_all])
            zs_t = torch.stack(zs_teacher_all)
            zs_sh = torch.stack(zs_shuf)

            # REPA (cosine): -mean(cos_sim) per depth → (n_depths,)
            zs_s_n = F.normalize(zs_s, dim=-1)
            zs_t_n = F.normalize(zs_t, dim=-1)
            zs_sh_n = F.normalize(zs_sh, dim=-1)
            repa_m = -(zs_s_n * zs_t_n).sum(dim=-1).mean(dim=(1, 2))
            repa_sh = -(zs_s_n * zs_sh_n).sum(dim=-1).mean(dim=(1, 2))

            # Smooth L1: per depth → (n_depths,)
            sl1_m = F.smooth_l1_loss(
                zs_s, zs_t, reduction="none"
            ).mean(dim=(1, 2, 3))
            sl1_sh = F.smooth_l1_loss(
                zs_s, zs_sh, reduction="none"
            ).mean(dim=(1, 2, 3))

            for depth_idx, d in enumerate(all_depths):
                info[f"diag/repa_matched_d{d}"] = repa_m[depth_idx].item()
                info[f"diag/repa_shuffled_d{d}"] = repa_sh[depth_idx].item()
                info[f"diag/repa_gap_d{d}"] = (repa_sh - repa_m)[depth_idx].item()
                info[f"diag/sl1_matched_d{d}"] = sl1_m[depth_idx].item()
                info[f"diag/sl1_shuffled_d{d}"] = sl1_sh[depth_idx].item()
                info[f"diag/sl1_gap_d{d}"] = (sl1_sh - sl1_m)[depth_idx].item()

    return teacher_loss, student_loss, projection_loss, info


def flow_reg_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
    cls_token: torch.Tensor,
    tgt_act_reps: torch.Tensor,
) -> float:
    """Flow model loss, matching the velocity field.

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        cls_token (torch.Tensor): the cls token
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.
        tgt_act_reps (torch.Tensor): target action representations

    Returns:
        float: the loss
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    cls_0 = torch.empty_like(cls_token[:, 0]).normal_(0, 1)  # NOTE: always uses cls_token from the first camera view
    cls_1 = cls_token[:, 0]

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)

    cls_t = interp.calc_It(t, cls_0, cls_1)
    cls_t_dot = interp.calc_It_dot(t, cls_0, cls_1)

    b_t, zs_tilde, b_t_cls = flow_map.get_velocity_reg(t, act_t, obs_emb, cls_t, align_depth=config.s_align_depth)

    if zs_tilde is None:
        raise ValueError(
            "zs_tilde is None — set network.align_depth to a value in [1, depth] to enable REPA"
        )

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)

    cls_loss = F.mse_loss(b_t_cls, cls_t_dot)
    cls_loss *= config.cls_loss_scale

    projection_loss = repa_loss(zs_tilde, torch.cat([cls_token.transpose(0, 1), tgt_act_reps], dim=2))
    projection_loss *= config.repa_scale
    return loss, projection_loss, cls_loss, {}


def regression_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Standard regression loss."""
    # sample
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    # compute loss
    loss = get_norm(act_pred - act, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def straight_flow_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Straight flow loss."""
    # sample
    t = torch.zeros_like(delta_t, device=delta_t.device)

    # Major difference compared to regression: use random noise instead of zeros
    act_0 = torch.randn_like(act, device=act.device)

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    # compute loss
    loss = get_norm(act_pred - act, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def tsd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Two step denoising loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.empty_like(act).normal_(0, 1)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss0 = get_norm((act_pred_0 - act_t) / config.t_two_step, config.norm_type)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def mip_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Minimum iterative policy loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # NOTE: in paper, we use
    # act_t = config.t_two_step * act + (1 - config.t_two_step) * noise
    # but we found that this is not necessary when config.t_two_step close to 1.
    # feel free to use the original form if you want to, you can refer to mip_origin_loss

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def mip_origin_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Minimum iterative policy loss (original form with scale in first iteration)."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = config.t_two_step * act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_target_0 = config.t_two_step * act
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    act_target_1 = act

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((act_pred_0 - act_target_0) / config.t_two_step, config.norm_type)
    loss1 = get_norm(
        (act_pred_1 - act_target_1) / (1 - config.t_two_step), config.norm_type
    )
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def lmd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Lagrangian map matching loss for distillation."""
    # sample
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    label = encoder(obs, None)

    # predict
    Is = interp.calc_It(s, act_0, act_1)
    Xst_Is, dt_Xst = flow_map.jvp_t(s, t, Is, label)

    # compute the target velocity field
    b_eval = flow_map.get_reference_velocity(t, Xst_Is, label)

    # lmd loss
    loss = torch.mean(
        (dt_Xst.flatten(start_dim=1) - b_eval.flatten(start_dim=1)) ** 2, dim=-1
    )
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def ctm_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Consistency trajectory model loss."""
    # sample
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    s_plus = s + config.discrete_dt
    t = torch.maximum(t, s_plus)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    Is = interp.calc_It(s, act_0, act_1)

    # compute the CTM loss
    Xst_Is_pred = flow_map(s, t, Is, obs_emb)
    b_s = flow_map.get_reference_velocity(s, Is, obs_emb)
    Is_plus = Is + config.discrete_dt * b_s
    Xst_Is_target = flow_map(s_plus, t, Is_plus, obs_emb)
    # make sure loss is not too small
    loss = config.loss_scale * torch.mean(
        ((Xst_Is_target - Xst_Is_pred) / config.discrete_dt) ** 2
    )

    return loss, {}


def psd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Progressive Self-Distillation loss combined with flow matching.

    This loss combines:
    1. Standard flow matching loss
    2. PSD term that encourages consistency between single-step and multi-step predictions

    The PSD term uses uniform weighting between intermediate steps.
    """
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== PSD term ==========
    # sample s, t, u like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # sample u uniformly between s and t
    h = torch.empty_like(delta_t).uniform_(0, 1)
    u = s + h * (t - s)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute full jump s -> t (student)
    _, f_xst = flow_map.get_map_and_velocity(s, t, Is, obs_emb)

    # compute two-step jump s -> u -> t (teacher, no stopgrad)
    xsu, f_xsu = flow_map.get_map_and_velocity(s, u, Is, obs_emb)
    _, f_xut = flow_map.get_map_and_velocity(u, t, xsu, obs_emb)

    # uniform PSD: teacher = (1 - h) * phi_su + h * phi_ut
    # where h is the relative position of u between s and t
    student = f_xst
    # expand h to match f_xsu dimensions: [batch, horizon, act_dim]
    h_expanded = h.view(-1, 1, 1)
    teacher = (1 - h_expanded) * f_xsu + h_expanded * f_xut

    # compute PSD loss using get_norm (ignore weight_st as requested)
    psd_term = get_norm(student - teacher, config.norm_type)
    psd_term = config.loss_scale * torch.mean(psd_term)

    # combine losses
    total_loss = flow_matching_loss + psd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "psd_term": psd_term.item(),
    }


def lsd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Lagrangian self-distillation loss combined with flow matching.

    This loss combines:
    1. Standard flow matching loss
    2. LSD term that encourages consistency in the velocity field

    The LSD term uses uniform sampling between s and t without stopgrad.
    """
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== LSD term ==========
    # sample s, t like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute Xst and dt_Xst using jvp_t
    xst, dt_xst = flow_map.jvp_t(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (no stopgrad)
    b_eval = flow_map.get_velocity(t, xst, obs_emb)

    # lsd loss (ignore weight_st)
    error = b_eval - dt_xst
    lsd_term = get_norm(error, config.norm_type)
    lsd_term = config.loss_scale * torch.mean(lsd_term)

    # combine losses
    total_loss = flow_matching_loss + lsd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "lsd_term": lsd_term.item(),
    }


def esd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Euler self-distillation loss."""
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== ESD term ==========
    # sample s, t like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute Xst and ds_Xst using jvp_t
    xst, ds_xst = flow_map.jvp_s(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (stopgrad)
    with torch.no_grad():
        b_eval = flow_map.get_velocity(t, xst, obs_emb)

    # compute jvp
    _, grad_xst_b = flow_map.jvp_x(s, t, Is, b_eval, obs_emb)

    # esd loss
    error = ds_xst + grad_xst_b
    esd_term = get_norm(error, config.norm_type)
    esd_term = config.loss_scale * torch.mean(esd_term)

    # combine losses
    total_loss = flow_matching_loss + esd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "esd_term": esd_term.item(),
    }


def mf_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Mean flow loss."""
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== Mean flow term ==========
    # sample s, t
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)
    dot_Is = interp.calc_It_dot(s, act_0, act_1)

    # compute Xst and ds_Xst using jvp_t
    xst, ds_xst = flow_map.jvp_s(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (stopgrad)
    with torch.no_grad():
        # Difference 1: use dot_Is instead of b_eval
        # Difference 2: also disable gradient for jvp_x
        # compute jvp
        _, grad_xst_b = flow_map.jvp_x(s, t, Is, dot_Is, obs_emb)

    # mf loss
    error = ds_xst + grad_xst_b
    mf_term = get_norm(error, config.norm_type)
    mf_term = config.loss_scale * torch.mean(mf_term)

    # combine losses
    total_loss = flow_matching_loss + mf_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "mf_term": mf_term.item(),
    }
