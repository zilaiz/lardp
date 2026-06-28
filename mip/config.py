from dataclasses import dataclass, field

from omegaconf import OmegaConf

if not OmegaConf.has_resolver("suffix"):
    OmegaConf.register_new_resolver("suffix", lambda s: f"_{s}" if s else "")


@dataclass
class LogConfig:
    log_dir: str
    wandb_mode: str
    project: str
    group: str
    exp_name: str
    exp_note: str = ""
    eval_freq: int = 20000
    log_freq: int = 1000
    save_freq: int = 10000
    eval_episodes: int = 10
    save_video: bool = False
    save_rollouts: bool = False
    rollout_noise_std: float = (
        0.0  # Gaussian noise std injected into actions during rollout collection
    )
    max_rollout_demos: int = 0  # Max rollout episodes to save (0 = unlimited)
    # Whether new-best models are copied to the repo-root ``checkpoints/`` dir
    # (the global, success-rate-named store). Off by default to avoid cluttering
    # checkpoints/ on sweeps; the per-run ``models/model_best.pt`` is still saved.
    # Set ``log.save_global_checkpoints=true`` for runs you want in the global store.
    save_global_checkpoints: bool = False


@dataclass
class OptimizationConfig:
    seed: int = 0
    loss_type: str = "flow"
    loss_scale: float = 100.0
    cls_loss_scale: float = 0.03
    repa_scale: float = 0.0  # REPA loss coefficient
    s_align_depth: int | list[int] = 2
    t_align_depth: int | list[int] = 2
    norm_type: str = "l2"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_steps: int = 1
    # Override the eval-time ODE step counts. None -> use
    # get_default_step_list(loss_type). Set e.g. [25] to eval at just 25 steps.
    eval_num_steps: list[int] | None = None
    sample_mode: str = "stochastic"  # "zero", "mean"
    # Goal-predictor ODE source override. None -> inherit `sample_mode`.
    # Useful when the goal distribution and the action distribution have
    # different modal structure (e.g. expert-only goals vs expert+play actions).
    goal_sample_mode: str | None = None
    t_two_step: float = 0.9
    discrete_dt: float = 0.01
    grad_clip_norm: float = 10.0
    ema_rate: float = 0.995
    batch_size: int = 1024
    gradient_steps: int = 300000
    warmup_ratio: float = 0.0
    rampup_ratio: float = 0.5
    min_value: float = 0.0
    max_value: float = 1.0
    model_path: str | None = None
    # Load encoder weights from a pretrained IDM checkpoint (ablation)
    pretrained_encoder_path: str | None = None
    freeze_encoder: bool = False
    interp_type: str = "linear"  # "linear" or "trig"
    device: str = "cuda"
    use_compile: bool = True  # Whether to use torch.compile for acceleration
    compile_mode: str = (
        "default"  # Compile mode: "default", "reduce-overhead", "max-autotune"
    )
    use_cudagraphs: bool = False  # Whether to use CUDA graphs (requires static shapes)
    auto_resume: bool = True  # Whether to automatically resume from checkpoint
    # IDM goal dropout (classifier-free guidance style)
    goal_dropout_prob: float = (
        0.0  # Probability of zeroing out goal frame during IDM training
    )
    # IDM encoder local-linearity regularizer weight (0 disables)
    local_linearity_coef: float = 0.0
    # Goal predictor specific
    idm_checkpoint_path: str | None = (
        None  # Path to pretrained IDM checkpoint (for goal predictor training)
    )
    state_matching_weight: float = (
        0.0  # Lambda for state-matching loss ||g_hat - encoder(s_{t+k})||^2
    )
    cfg_scale: float = 1.0  # CFG scale for goal predictor inference (1.0 = no guidance)
    # Goal predictor DiT specific
    goal_flow_num_steps: int = 5  # ODE steps for goal generation at inference
    goal_flow_loss_scale: float = 1.0  # weight for state flow loss
    action_reg_weight: float = 1.0  # weight for action regularization loss
    action_reg_t_weighting: str = (
        "linear"  # per-sample weighting of action loss by t_flow: "none" | "linear"
    )
    action_reg_num_steps: int = 1  # K Euler steps from x_t -> g_hat for action_reg_loss; 1 = original one-step shortcut
    goal_stats_path: str | None = (
        None  # path to precomputed goal normalization stats (.pt)
    )
    delta_stats_path: str | None = (
        None  # path to precomputed delta (= z_goal - z_last_obs) stats (.pt) for the delta predictor
    )
    # Goal predictor DDT-NS (noise-shift) variant only. SD3-style time shift on
    # the single goal flow time, plus a configurable base-t distribution and
    # endpoint clamp. Identity defaults so leaving them untouched recovers the
    # plain GP behavior.
    goal_t_shift: float = 1.0  # SD3 shift t' = a*t / (1 + (a-1)*t); 1.0 = no shift
    goal_t_dist: str = "uniform"  # "uniform" | "logit_normal"
    goal_t_dist_mu: float = 0.0  # logit_normal mean (sigmoid(N(mu, sigma)))
    goal_t_dist_sigma: float = 1.0  # logit_normal std
    goal_t_eps: float = 0.0  # clamp t to [eps, 1-eps] at training + inference
    # Policy DDT-NS (noise-shift) variant only — used by ``flow_ns`` loss +
    # ``ode_ns_sampler`` to apply SD3 time shift on the single action flow
    # time. Identity defaults recover plain flow matching.
    policy_t_shift: float = 1.0  # SD3 shift t' = a*t / (1 + (a-1)*t); 1.0 = no shift
    policy_t_dist: str = "uniform"  # "uniform" | "logit_normal"
    policy_t_dist_mu: float = 0.0
    policy_t_dist_sigma: float = 1.0
    policy_t_eps: float = 0.0  # clamp t to [eps, 1-eps] at training + inference
    # Goal predictor o2g consistency loss (0 disables; mirrors VITA's FLC but with
    # action FM as the comparator rather than direct latent MSE)
    consistency_weight: float = 0.0
    consistency_num_steps: int | None = None  # None -> use goal_flow_num_steps
    # Goal predictor regressor: deterministic L2 + action-FM training
    goal_l2_weight: float = 1.0  # weight for L2 loss on expert goal embedding
    # Whether the L2 loss is computed in normalized or raw encoder space. With
    # no goal_stats_path, "normalized" auto-falls-back to "raw"; default "raw"
    # keeps the no-stats path silent (the recommended default).
    goal_l2_target_space: str = "raw"
    # Goal predictor retrieval: non-parametric, looks up nearest training goal.
    retrieval_index_path: str | None = None  # path to (.pt) index built offline
    retrieval_query_space: str = "obs_summary"  # "flatten_zt" | "obs_summary"
    retrieval_distance: str = "l2"  # "l2" | "cosine"
    retrieval_top_k: int = 1
    retrieval_aggregation: str = "top1"  # "top1" | "weighted"
    retrieval_temperature: float = 1.0  # softmax temperature for "weighted"
    # If True, retrieve both obs_summary (= index keys) and goal embedding
    # (= index values) from the nearest training row, and inject the pair
    # directly into the IDM action trunk via ``forward_with_summary``,
    # bypassing the IDM's obs_summarizer. Forces top1 and requires
    # query_space="obs_summary" + a v2 IDM with ``forward_with_summary``.
    retrieval_use_retrieved_pair: bool = False
    # BYOL-FDM proprioception reconstruction auxiliary (FDMAgent).
    # When > 0, FDMNet's proprio_decoder predicts the per-frame low-dim obs
    # (proprioception) from the encoded latent. Acts as a non-collapsible
    # supervised signal on the encoder, mirroring TDMPC2's use of supervised
    # heads to prevent representational collapse during pretraining.
    proprio_recon_loss_scale: float = 0.0  # 0 = disabled
    # Downstream loading knob (LBMDiTJointDDTFrozenFDMAgent only): if True,
    # initialize the input encoder from the BYOL EMA target encoder
    # (state_dict["encoder_ema"]) instead of the online encoder
    # (state_dict["encoder"]). For BYOL/SPR-style pretraining the EMA target
    # is sometimes the smoother / preferred downstream artifact.
    joint_use_encoder_ema_for_init: bool = False
    # IDM + FDM joint training (lbmidm_v2 / IDMFDMAgent)
    fdm_loss_scale: float = (
        0.0  # weight for forward-dynamics auxiliary loss (0 = disabled)
    )
    # Ortho regularizer: hinge-form penalty on cos_sim(z_obs[-1], z_goal),
    # active only when cos_sim > ortho_reg_threshold. The hinge gives a
    # stable equilibrium (no runaway orthogonality) and avoids the
    # FDM-vs-ortho phase transition observed with no-threshold cos_sim
    # minimization. Both inputs are raw encoder outputs (no MLP in between),
    # so the regularizer shapes the encoder directly. 0 disables.
    ortho_reg_weight: float = 0.0
    ortho_reg_threshold: float = 0.5
    # Dropout annealing for condistill extra_cond_encoder
    extra_cond_dropout_warmup_steps: int = 5000  # number of steps to keep dropout at 0
    extra_cond_dropout_rampup_steps: int = (
        10000  # number of steps to linearly ramp dropout from 0 to max
    )
    # Diagnostic: measure how much extra_cond affects teacher representations
    diagnose_teacher_delta: bool = False
    # LBMDiTJoint (single-stage joint state+action DiT) -------------------
    # Per-stream loss weights (final loss = state_w * L_state + action_w * L_act).
    joint_state_loss_weight: float = 1.0
    joint_action_loss_weight: float = 1.0
    # CFG dropout: fraction of training steps where the optimality label is
    # overridden with the null/play index (slot 1). Default 0.0 because play
    # data already trains the null slot directly; raise only if you want extra
    # regularization on the null embedding.
    joint_cfg_dropout_prob: float = 0.0
    # Inference-time CFG strength `w` in v_guided = (1+w)*v_cond - w*v_uncond.
    # 0 = plain conditional sampling (one network call per ODE step).
    joint_cfg_scale: float = 0.0
    # Whether to condition the trunk on the optimality (expert/play) label.
    # True (default): the net receives the source label and CFG applies — the
    # current behavior. False: RETIRE the optimality conditioning — the net
    # gets optimality_idx=None (constant null embedding), CFG dropout and
    # inference-time CFG are both disabled, so the expert/play separation rests
    # entirely on the LOSS GATING (e.g. joint_play_scheme=region's tau gates).
    # The true source label is still consumed for that loss masking; this knob
    # only controls whether it also drives the network's conditioning. Used to
    # test whether geometric region-gating alone can replace the learned label.
    joint_use_optimality: bool = True
    # ODE source for sampling (state/action streams share the same mode).
    # "stochastic" = Gaussian noise, "zero" = deterministic.
    joint_sample_mode: str = "stochastic"
    # Number of Euler steps for joint ODE sampling at inference.
    joint_num_steps: int = 5
    # Whether the encoder loaded from idm_checkpoint_path is frozen (default) or
    # fine-tuned alongside the joint trunk.
    joint_freeze_encoder: bool = True
    # LBMDiTJointDDTFrozenAgent variant: if True, the input/conditioning encoder
    # is initialized from the IDM checkpoint and fine-tuned alongside the joint
    # trunk, while a second *frozen* copy of the encoder (snapshot of the same
    # IDM weights) is used to compute the next-state embedding target. This
    # gives the input pathway gradient flow without making the FM target a
    # moving target. Overrides ``joint_freeze_encoder`` when True.
    joint_finetune_input_encoder: bool = False
    # DP-pretrained encoder source (LBMDiTJointDDTFrozenDPAgent variant).
    # Path to a pretrained LBMDiT/DP checkpoint whose encoder weights will be
    # loaded into the joint trunk's obs encoder instead of an IDM-pretrained
    # one. Mutually exclusive with idm_checkpoint_path for that agent.
    dp_checkpoint_path: str | None = None
    # If True, load weights from ``encoder_ema`` rather than ``encoder`` —
    # the smoother choice for downstream feature use. Defaults to True.
    dp_use_encoder_ema: bool = True
    # Frozen-target ablation only (LBMDiTJointPTFrozenTargetAgent): also warm-
    # start the LIVE input encoder from the SAME DP checkpoint used for the
    # frozen target (``dp_checkpoint_path``, key chosen by ``dp_use_encoder_ema``),
    # so the condition encoder and the frozen target start in the SAME
    # representation space. The input encoder stays trainable (finetuned); only
    # the target stays frozen. Isolates whether s2e=True still degrades the
    # policy when there is no foreign-manifold gap to cross at init. Requires
    # ``idm_checkpoint_path`` to be null (the two are competing input-encoder
    # warm-starts). Default False = scratch / idm-warm-start as before.
    init_input_encoder_from_dp: bool = False
    # Frozen-target ablation (LBMDiTJointPTFrozenTargetAgent): which pluggable
    # TargetEncoder produces the FM state-flow target (see mip/target_encoders.py).
    # "dp"  -> a frozen DP/LBMDiT MultiImageObsEncoder loaded from
    #          dp_checkpoint_path (current behavior). Foreign pretrained visual
    #          encoders ("lewm", "dinov2", ...) are added incrementally behind
    #          the same interface. The per-dim z-score (goal_stats_path) is
    #          applied on top regardless of source.
    target_encoder_type: str = "dp"
    # Frozen-target ablation: checkpoint path/dir for non-DP target encoders
    # (e.g. the LeWM HF ViT dir containing weights.pt + config.json). The DP
    # target uses dp_checkpoint_path; foreign encoders use this.
    target_encoder_path: str | None = None
    # Which rgb obs key the (image-only) foreign target encoder consumes. None
    # = auto-pick the single rgb key in task.shape_meta (errors if 0 or >1, e.g.
    # multi-camera robomimic — set it explicitly there). pusht -> "image".
    target_encoder_image_key: str | None = None
    # E2E variant only (LBMDiTJointE2EAgent): whether the target LayerNorm
    # has learnable gamma/beta. Default False removes the gamma->0 collapse
    # mode; flip to True for the UNITE-faithful variant (encoder gets more
    # expressive target normalization at the cost of a real collapse path).
    joint_target_ln_affine: bool = False
    # Whether to keep the LayerNorm (``target_ln``) on the INPUT encoder's
    # output before it becomes the AdaLN condition. Default True = current
    # behavior. False feeds the RAW encoder output as the condition
    # (``target_ln`` becomes ``nn.Identity`` — no params, so optimizer / EMA /
    # save-load all no-op, and every ``target_ln(...)`` call site, train and
    # eval, passes through). Intended for the FROZEN-TARGET ablation
    # (LBMDiTJointPTFrozenTargetAgent), where ``target_ln`` is PURELY the
    # condition normalization (the FM target is a frozen external encoder
    # z-scored by precomputed stats, not target_ln'd). It thus cleanly toggles
    # "per-sample LN'd condition" vs "raw condition" to probe how much of the
    # s2e=True degradation is the condition/target normalization-space mismatch.
    # CAVEAT for self-referential agents (E2E/DDT/PT without a frozen target):
    # the default ``_encode_condition_target`` applies ``target_ln`` to BOTH
    # the condition AND the FM target, so setting this False there also removes
    # the target-side normalization (a separate, less-stable config) — it is
    # meant to be flipped only on the frozen-target agent.
    joint_input_ln: bool = True
    # E2E variant only (LBMDiTJointE2EAgent / DDT): use the EMA encoder +
    # EMA target_ln to compute the FM state target during training. This
    # decouples the regression target from per-step encoder updates, the
    # standard self-distillation fix (BYOL/DINO/REPA-E) for the moving-
    # target dynamic where state_loss creeps up as the encoder evolves.
    # Reuses ``ema_rate`` for the EMA decay. Active only when ``ema_rate
    # < 1``; otherwise falls through to the live encoder. Default False
    # preserves baseline behavior (target from live encoder).
    joint_use_ema_target: bool = False
    # E2E/DDT variant only: if False, block state_loss gradients from
    # reaching the encoder and target_ln by running a second trunk forward
    # pass with z_t.detach() to produce v_state. action_loss still flows
    # into the encoder via the live z_t forward used for v_action. Costs
    # ~2x trunk compute. Motivation: state_loss creates a self-distillation
    # loop on the shared representation (encoder is both the source of
    # the regression target and the conditioning input), which can fight
    # action_loss and destabilize encoder training. Default True preserves
    # baseline behavior (single forward, state_loss shapes encoder).
    joint_state_loss_to_encoder: bool = True
    # Companion routing knob for the ACTION/IDM loss (mirror of the state flag
    # above). Default True = baseline: the action loss always reaches the encoder
    # via the LIVE condition (the collapse-free external-action-target anchor).
    # Set False to detach the action head's condition so the action loss does NOT
    # shape the encoder. Together the two flags span the 2x2 of which losses
    # shape the input encoder:
    #   (state F, action T) = action-only  (recommended baseline; old s2e=False)
    #   (state T, action T) = both         (old s2e=True; warned ablation)
    #   (state T, action F) = state-only   (FDM-only encoder; the new cell —
    #                         tests whether next-state prediction ALONE yields an
    #                         action-sufficient encoder, i.e. retention not just
    #                         geometry)
    #   (state F, action F) = neither      (encoder gets no live grad; degenerate)
    # update() implements this by feeding each head a live or detached condition:
    # ONE forward when both heads share a condition, TWO when they differ.
    joint_action_loss_to_encoder: bool = True
    # E2E/DDT/PT variant only: LR multiplier for the encoder + target_ln
    # param group, relative to the trunk LR (``lr``). 1.0 = single group
    # (baseline, exact checkpoint/optimizer compat). Values < 1 slow the
    # representation relative to the trunk so the trunk adapts to a
    # slowly-moving latent rather than the latent collapsing to ease the
    # denoising objective (JEDI uses 0.3; spirit of JEPA/TD-MPC2). Anti-
    # collapse lever for when state_loss is allowed into the encoder.
    joint_encoder_lr_scale: float = 1.0
    # DDT trunk ablation: when True, replace the trunk's x_state input with
    # a learnable global state token. t_state still passes through from the
    # caller, so the state slot's AdaLN cond is exercised across the full
    # t-range (independently sampled when joint_decouple_t=True, shared
    # with t_action when False). Isolates the contribution of per-sample
    # state generation to action denoising: trunk structure, attention
    # paths, parameter count, and both heads are preserved, but the state
    # slot carries no per-sample information. Hard-requires
    # joint_state_loss_weight == 0 (the state head is asked to predict
    # velocity from a constant input under a zeroed loss, which is only
    # coherent when no gradient flows through v_state). The trunk's
    # `learnable_state_token` is trained via the action_loss gradient (a
    # global learnable bias on the state slot).
    joint_replace_x_state: bool = False
    # Mixed-data sampling (joint pipeline, expert + IDM rollouts):
    # Target fraction of each training batch drawn from the expert (primary)
    # dataset; rollouts get (1 - fraction). Implemented via WeightedRandomSampler
    # over the ConcatDataset, so the natural size imbalance between expert and
    # rollouts doesn't dilute expert-conditional updates. None disables the
    # reweighting (falls back to uniform shuffle = natural proportions). No-op
    # when the dataset is a single source (no rollouts).
    expert_sample_fraction: float | None = 0.5
    # Vanilla LBMDiT optimality conditioning (network.use_optimality=True):
    # CFG dropout fraction — randomly relabel expert->null during training so
    # the null slot trains as an unconditional reference. 0 = off (play data
    # already trains the null slot directly). Mirrors joint_cfg_dropout_prob
    # but for the plain DP path. At inference the policy conditions on the
    # expert slot (EXPERT_IDX). No-op when use_optimality=False.
    opt_cfg_dropout_prob: float = 0.0
    # DDT/decoupled-time variant only (LBMDiTJointDDTAgent):
    # If True, sample independent t_state and t_action per batch during
    # training (DF-style decoupled noising). If False, the same scalar t
    # is used for both streams (diagonal training, matches LBMDiTJointE2E).
    joint_decouple_t: bool = True
    # Play-data ablation: when True, exclude the both-near-noise corner of the
    # decoupled (t_state, t_action) square for play-source samples (optimality
    # == NULL pre-CFG-dropout). That corner = unconditional joint generation of
    # a (suboptimal) rollout, which you want only on expert data; play keeps
    # the IDM/FDM-like regimes. If both base times < tau, one randomly chosen
    # stream is lifted into [tau, hi]. No-op when joint_decouple_t=False.
    joint_play_avoid_both_noise: bool = False
    # Threshold (pre-shift base-t space) defining "near the noise end" for the
    # above: a play sample is in the avoided corner when both base t < tau.
    joint_play_both_noise_tau: float = 0.5
    # Loss-assignment scheme over the decoupled (t_state, t_action) square.
    #   "legacy":       both losses on every row (baseline behavior).
    #   "noisier_all":  every row contributes only the loss of its noisier
    #                   stream (lower t): predict the noisier stream from the
    #                   strictly cleaner one. The diagonal partitions the
    #                   square into a continuous IDM-like half (action loss,
    #                   cleaner state) and an FDM-like half (state loss,
    #                   cleaner actions). Applies to ALL data sources — the
    #                   expert policy objective changes too (each expert row
    #                   carries one loss instead of two).
    #   "noisier_play": the rule above for play-source rows only (optimality
    #                   == NULL pre-CFG-dropout); expert rows keep both
    #                   losses, so the expert policy objective is identical
    #                   to legacy and play adds continuous-spectrum dynamics
    #                   supervision on top.
    # Ties (t_state == t_action; measure-zero when decoupled) take the action
    # loss. Non-legacy schemes require joint_decouple_t=True. They are
    # routing-agnostic: with joint_state_loss_to_encoder=False (recommended;
    # only the action loss shapes the encoder, so rule-governed rows on the
    # state-noisier side contribute no encoder gradient), True is a warned
    # ablation that lets active state-loss rows shape the encoder.
    # Corner-avoidance (joint_play_avoid_both_noise) is inert under
    # non-legacy schemes.
    #   "region":       per-stream honesty gate over the (t_state, t_action)
    #                   square. Each stream is compared to its OWN threshold
    #                   (post-shift):
    #                       s_clean = t_state  >= joint_region_tau_state
    #                       a_clean = t_action >= joint_region_tau_action
    #                   EXPERT optimizes BOTH losses everywhere: for expert no
    #                   region is contaminating or harmful (at worst low-signal
    #                   when predicting the already-clean stream), so dropping
    #                   any of it would only waste the scarce expert set.
    #                   ROLLOUT optimizes a loss only where it is honest
    #                   dynamics, i.e. that loss's CONDITIONER stream is clean:
    #                     action loss (IDM) needs s_clean (next-state known);
    #                     state  loss (FDM) needs a_clean (action known).
    #                   So over the cells, rollout supplies:
    #                     corner ~s&~a : neither (policy+plan = expert only)
    #                     UL      s&~a : action loss only (IDM)
    #                     LR     ~s& a : state loss only (FDM)
    #                     TR      s& a : both (both honest dynamics)
    #                   Requires joint_decouple_t=True.
    joint_play_scheme: str = "legacy"
    # "region" scheme thresholds (post-shift time). A stream is "clean enough"
    # to be an honest dynamics conditioner once its time is >= its threshold;
    # below it, rollout predicting the OTHER stream would be policy/plan
    # (optimality-laden), so only expert supplies it there. tau_state gates the
    # ROLLOUT action loss (how known the next-state must be for IDM); tau_action
    # gates the ROLLOUT state loss (how known the action must be for FDM). They
    # may differ — inverse and forward dynamics have different identifiability.
    # Expert is unaffected by these (it optimizes both losses everywhere).
    joint_region_tau_state: float = 0.5
    joint_region_tau_action: float = 0.5
    # Inference t-schedule for the (state, action) flow pair.
    #   "diagonal":    t_state = t_action = grid (single Euler walk).
    #   "state_first": clean state first (t_state ramps 0->1 in first half),
    #                  then clean action (t_action ramps 0->1 in second half).
    #   "pyramid":     t_state leads t_action by a fixed offset throughout.
    #   "action_only": t_state pinned at eps; only action walks. Ablation
    #                  for the value of joint state denoising; requires
    #                  joint_sample_mode == "stochastic" to stay in-dist.
    joint_t_schedule: str = "diagonal"
    # Pyramid offset (only used when joint_t_schedule == "pyramid"). Positive
    # value means t_state advances ahead of t_action by this much (fraction
    # of the [0, 1] range). Clamped to [0, 1] at each step.
    joint_pyramid_offset: float = 0.2
    # Small epsilon to keep schedule endpoints away from {0, 1} (avoids OOD
    # at the data/noise corners). Applied symmetrically: schedule lives in
    # [eps, 1-eps]. 0 disables.
    joint_t_eps: float = 0.0
    # Per-stream SD3-style time shift. Formula: t' = a*t / (1 + (a-1)*t).
    # In mip's convention (t=0 noise, t=1 data): a < 1 emphasizes the
    # noisy regime (UNITE setting), a > 1 emphasizes the data regime.
    # Default 1.0 (no shift) preserves baseline behavior. Applied to both
    # training-time t sampling and inference-time t grids on the matching
    # stream — DO NOT change one without the other.
    joint_t_shift_state: float = 1.0
    joint_t_shift_action: float = 1.0
    # Base distribution used to sample training t before the per-stream
    # shift is applied. "uniform" uses Uniform[eps, 1-eps]; "logit_normal"
    # uses sigmoid(N(mu, sigma)) clamped to [eps, 1-eps], matching SD3/UNITE.
    # "beta" mirrors flow_beta_loss / PI-0: u ~ Beta(1.5, 1.0), t = 0.999*(1-u);
    # biases mass toward the noise end (mip's t=0). eps is ignored (0.999 cap
    # is intrinsic); leave joint_t_shift_* at 1.0 to match PI-0 exactly.
    # "reverse_beta" reproduces the original (pre-fix) flow_beta: t ~ Beta(1.5,
    # 1.0) directly, mass at t≈1 (data end). For A/B comparisons only.
    joint_t_dist: str = (
        "uniform"  # "uniform" | "logit_normal" | "beta" | "reverse_beta"
    )
    joint_t_dist_mu: float = 0.0
    joint_t_dist_sigma: float = 1.0
    # Optional per-stream overrides of the base t distribution (only used when
    # joint_decouple_t=True). Each defaults to None -> fall back to the shared
    # joint_t_dist / mu / sigma above, so existing configs are unchanged. Lets
    # the state and action streams train on different t schedules, e.g. state =
    # logit_normal + shift (UNITE, pairs with joint_state_param="x1") while
    # action = beta unshifted (PI-0, pairs with joint_action_param="velocity").
    # mu / sigma only matter for "logit_normal"; the beta branches ignore them.
    joint_t_dist_state: str | None = None
    joint_t_dist_mu_state: float | None = None
    joint_t_dist_sigma_state: float | None = None
    joint_t_dist_action: str | None = None
    joint_t_dist_mu_action: float | None = None
    joint_t_dist_sigma_action: float | None = None
    # State-head parameterization (DDT/PT joint agents).
    #   "velocity": network's state-head output is the velocity v_state; loss
    #       is MSE(v_state - s_t_dot). Baseline / backwards-compatible.
    #   "x1":      UNITE-style. The raw network output is treated as the
    #       x1-estimate s_pred of the (LN'd) goal-obs embedding — no LN is
    #       applied to the prediction itself; it is regressed against the
    #       LN'd target. v_state = (s_pred - s_t) / max(1 - t_state, eps)
    #       is derived analytically for both the loss and Euler integration,
    #       with the same clamped denominator applied to v_gt (so the loss
    #       is an unbiased x1-MSE with a 1/(1-t)^2 weighting capped at
    #       1/eps^2). At sampling, CFG mixes the raw s_pred per-branch,
    #       then v_state is derived.
    joint_state_param: str = "velocity"
    # Action-head parameterization (DDT/PT joint agents).
    #   "velocity": network's action-head output is v_action; loss is
    #               MSE(v_action - a_t_dot). Baseline / backwards-compatible.
    #   "x1":       Network output is treated as the x1-estimate ``a_pred`` of
    #               the clean action chunk; v_action is derived as
    #               (a_pred - a_t) / (1 - t_action) for both loss and Euler
    #               integration. SAME clamped denominator is applied to v_gt.
    #               NO LN is applied — actions aren't a learned representation
    #               on a target manifold, so UNITE's LN-on-output trick doesn't
    #               transfer cleanly. Loss form stays velocity-MSE (equivalent
    #               to x1-MSE with 1/(1-t)^2 weighting; matches PI-0 /
    #               standard FM-policy parameterization).
    joint_action_param: str = "velocity"
    # Floor on (1 - t) when deriving v from x1-prediction under the x1
    # parameterization. Mirrors UNITE's train_eps/sample_eps=5e-2 clamp; used
    # by both state and action streams when their respective param is "x1".
    joint_x1_pred_eps: float = 5e-2


@dataclass
class NetworkConfig:
    network_type: str = "mlp"  # "mlp" or "cnn"
    num_layers: int = 4
    emb_dim: int = 512
    dropout: float = 0.1
    encoder_dropout: float = 0.0
    encoder_type: str | None = (
        None  # "mlp", "per_step_mlp", "identity", "image", "dino"
    )
    extra_cond_encoder_dropout: float = 0.4
    expansion_factor: int = 4
    timestep_emb_dim: int = 128
    timestep_emb_type: str = "positional"  # Type of timestep embedding
    # State encoder configs
    num_encoder_layers: int = 2  # Number of layers for MLP encoder
    # Image encoder configs
    rgb_model_name: str = "resnet18"
    use_seq: bool = True
    keep_horizon_dims: bool = True
    # Transformer specific configs
    n_heads: int = 6
    n_cond_layers: int = 0
    attn_dropout: float = 0.1
    # UNet specific configs
    model_dim: int = 256
    kernel_size: int = 5
    cond_predict_scale: bool = True
    obs_as_global_cond: bool = True
    dim_mult: list[int] | None = None
    norm_type: str = "groupnorm"
    attention: bool = False
    # RNN specific configs
    rnn_type: str = "LSTM"  # "LSTM" or "GRU"
    max_freq: float = 100.0
    # Encoder output dimension override (None = use emb_dim)
    encoder_out_dim: int | None = None
    # Joint trunk only: decouple the STATE-stream (FM target) dim from the
    # condition/obs dim. None = use obs_dim (encoder_out_dim or emb_dim), the
    # current coupled behavior. Set to the target encoder's output dim (e.g.
    # 192 for a LeWM CLS+projector target) when denoising toward a foreign
    # representation whose dim differs from the input encoder's.
    state_target_dim: int | None = None
    # Frozen pretrained ViT encoder (encoder_type="frozen_vit"): a frozen
    # backbone (DINOv2/SigLIP) + a trainable per-camera attentive-pool (MAP)
    # adapter over its patch tokens, concatenated with low-dim state and
    # projected to encoder_out_dim. Shared for the AdaLN condition and the FM
    # state target in the joint_pt pipeline (the target uses the EMA adapter;
    # the frozen backbone is shared, so the target is anchored to fixed
    # pretrained features and cannot collapse).
    frozen_vit_backbone: str = "dinov2"  # "dinov2" | "siglip"
    frozen_vit_path: str | None = None  # local HF model dir (required)
    frozen_vit_n_query: int = 1  # MAP probes per camera view
    frozen_vit_n_heads: int = 8  # MAP attention heads
    # How each camera view's patch tokens become the per-view feature fed to the
    # fusion MLP:
    #   "map"    : trainable per-view AttentivePool (MAP) head over the patch
    #              tokens (default; n_query/n_heads above apply).
    #   "frozen" : use the backbone's NATIVE pooled descriptor (DINOv2 CLS /
    #              SigLIP pooler_output) directly — NO trainable per-view adapter,
    #              so the fusion MLP over concat(views, proprio) is the only
    #              trainable image-side capacity. n_query/n_heads are ignored, and
    #              the backbone is forced to load with_pooled=True (keeps SigLIP's
    #              pooler head; no-op for DINOv2).
    frozen_vit_input_pool: str = "map"  # "map" | "frozen"
    # bf16-autocast the frozen backbone forward on CUDA (frozen + no-grad, so a
    # safe ~2x speedup; the trainable MAP head stays fp32). Disable for exact
    # fp32 features.
    frozen_vit_autocast_bf16: bool = True
    # Cap images per backbone forward to bound peak activation memory (null /
    # <=0 = one forward over all cameras*frames). Raise headroom for big
    # backbones / many cameras (e.g. transport) / large batch; no-op when large.
    frozen_vit_backbone_chunk_size: int | None = None
    # FM state TARGET proprio toggle (frozen_vit only). The condition always
    # includes low-dim proprio; when False the target zeros the low-dim slot so
    # the state stream denoises toward a pure visual next-state. Default True =
    # parity with the standard joint_pt target (proprio in the target).
    target_include_proprio: bool = True
    # Expose the frozen backbone's NATIVE pooled descriptor (DINOv2 CLS /
    # SigLIP pooler_output) so it can be reused as the FM state target by the
    # shared-backbone frozen-target ablation (LBMDiTJointPTFrozenViTTargetAgent).
    # For SigLIP this keeps the pretraining MAP pooler head (otherwise dropped
    # via vision_use_head=False); for DINOv2 the CLS is always present, so this
    # is a no-op there. The input MAP-adapter path never consumes the pooled
    # descriptor, so leaving this on for the plain frozen_vit input encoder is
    # harmless (the head is frozen and stripped from the checkpoint).
    frozen_vit_expose_pooled: bool = False
    # Vanilla LBMDiT optional optimality conditioning (expert=0 / play=1),
    # mirroring the joint trunks' optimality embedding but for the plain DP
    # DiT. Off by default = byte-identical to the original LBMDiT (no extra
    # params, loads old checkpoints unchanged). Module names match
    # franka_diff's LBMDiT so checkpoints stay deploy-compatible. Consumed by
    # TrainingAgent only when ``use_optimality=True``; reuses the same
    # ``optimization.expert_sample_fraction`` / ``opt_cfg_dropout_prob`` knobs.
    use_optimality: bool = False
    opt_emb_dim: int | None = None  # optimality embedding width (None -> d_model)
    opt_cond_compose: str = "add"  # "add" (into time features) | "concat" (append)
    # REPA specific
    projector_dim: int = 2048
    z_dims: list[int] | None = None
    # Goal predictor MLP architecture
    goal_predictor_hidden_dims: list[int] | None = None  # e.g., [512, 512]
    goal_predictor_dropout: float = 0.1
    # Goal predictor DiT architecture
    goal_dit_depth: int = 6
    goal_dit_n_heads: int = 6
    goal_dit_d_model: int | None = None  # None = use emb_dim
    goal_dit_dropout: float = 0.1
    goal_dit_projector_dim: int | None = None  # None = 2 * d_model
    goal_dit_timestep_emb_dim: int | None = None  # None = use goal_dit_d_model
    goal_align_depth: int | list[int] | None = 3
    # Goal predictor DiT with DDT head (encoder-decoder split)
    goal_ddt_enc_depth: int = 8
    goal_ddt_dec_depth: int = 2
    goal_ddt_d_model_enc: int | None = None  # None = use emb_dim
    goal_ddt_d_model_dec: int | None = None  # None = same as d_model_enc
    goal_ddt_n_heads_enc: int = 8
    goal_ddt_n_heads_dec: int = 8
    goal_ddt_dropout: float = 0.0
    # DDT-NS variant: decouple time embedding width from d_model_enc (None -> d_model_enc)
    goal_ddt_timestep_emb_dim: int | None = None
    # LBMDiTDDT (action-only DDT policy trunk; encoder/decoder width split,
    # global cond, single t — paired with ``flow_ns`` loss for noise-shift A/Bs).
    policy_ddt_enc_depth: int = 4
    policy_ddt_dec_depth: int = 2
    policy_ddt_d_model_enc: int | None = None  # None -> emb_dim
    policy_ddt_d_model_dec: int | None = None  # None -> 2 * d_model_enc
    policy_ddt_n_heads_enc: int = 8
    policy_ddt_n_heads_dec: int = 8
    policy_ddt_timestep_emb_dim: int | None = None  # None -> d_model_enc
    # LBMDiTIDMv2 (IDM with obs summarizer + FDM head)
    obs_summarizer_hidden: int | None = None  # None -> 2 * obs_dim
    action_proj_hidden: int | None = None  # None -> 2 * obs_dim
    fdm_hidden: int | None = None  # None -> 2 * obs_dim
    # LBMDiTJoint (single-stage joint state+action DiT)
    joint_opt_emb_dim: int | None = None  # None -> obs_dim
    # Per-token cond composition style for the joint DDT/PT trunks.
    # "add" sums (time + obs + opt) at trunk width (current default).
    # "concat" stacks them along the feature axis, growing the cond width
    # by 3x and letting the AdaLN modulation linear learn the mixing
    # weights instead of receiving a fixed-coefficient sum.
    joint_cond_compose: str = "add"  # "add" | "concat"
    # Decouple the state vs action streams INSIDE each joint-PT transformer
    # block: when True, the state token and the action tokens get SEPARATE AdaLN
    # modulation AND SEPARATE MLP weights, so the next-state (FDM) and action
    # (IDM) streams are processed independently within the block. Self-attention
    # stays shared (the joint mixing op); the cond front-end (time/obs/opt MLPs)
    # stays shared; the output heads were already separate. False (default): one
    # shared modulation + one shared MLP per block (original; checkpoints load
    # unchanged). Used to test whether the state/action coupling inside the block
    # drives the state_loss_to_encoder=True harm. LBMDiTJointPT trunk only;
    # ~+1 modulation MLP and +1 MLP per block in params.
    joint_decouple_streams: bool = False
    # LBMDiTJointDDT (joint DiT with encoder/decoder width split)
    joint_ddt_enc_depth: int = 8
    joint_ddt_dec_depth: int = 2
    joint_ddt_d_model_enc: int | None = None  # None -> emb_dim
    joint_ddt_d_model_dec: int | None = None  # None -> 2 * d_model_enc
    joint_ddt_n_heads_enc: int = 8
    joint_ddt_n_heads_dec: int = 8
    # Goal predictor o2g (obs-summary -> goal embedding, VITA-style residual MLP)
    o2g_hidden_dim: int | None = None  # None -> 2 * obs_dim
    o2g_num_layers: int = 4
    o2g_mlp_ratio: int = 4
    o2g_dropout: float = 0.0
    o2g_timestep_emb_dim: int | None = None  # None -> obs_dim
    o2g_cond_mode: str = "none"  # "none" | "obs_summary"
    # Goal predictor regressor (deterministic transformer w/ goal-query token)
    regressor_depth: int = 6
    regressor_n_heads: int = 6
    regressor_d_model: int | None = None  # None = use emb_dim
    regressor_dropout: float = 0.0
    regressor_mlp_ratio: int = 4


@dataclass
class TaskConfig:
    env_name: str = "lift"
    obs_type: str = "state"
    env_type: str = "ph"
    abs_action: bool = True
    action_type: str = "absolute"  # "absolute" or "delta"
    # When set, transform absolute actions into chunk-relative deltas at
    # dataset-load time. Currently the only supported value is "current_obs":
    # anchor = robot0_eef_pos/quat at the LAST obs frame, applied across the
    # full action chunk (matches openpi/pi 0.5 + DROID/robosuite convention).
    # The transformed action is 7-dim [pos_delta(3), axis_angle_delta(3), grip(1)].
    # See mip/franka_delta_transform.py for the math.
    delta_action_anchor: str | None = None
    # Normalizer for the (delta) action channel: "quantile" (default; maps the
    # 1st/99th percentile per dim to [-1, 1], robust to outliers — recommended
    # for delta actions whose rotation channel is tiny + heavy-tailed) or
    # "minmax" (legacy global min/max, the DP default). Only affects the action
    # normalizer; obs normalizers stay MinMax. Absolute-action runs keep MinMax
    # regardless (the abs channels are full-range, so the choice barely matters).
    delta_action_normalizer: str = "quantile"
    # Dataset configuration - either HuggingFace or local path
    dataset_repo: str | None = (
        None  # HuggingFace repository ID (e.g., "ChaoyiPan/mip-dataset")
    )
    dataset_filename: str | None = (
        None  # Path within the repository (e.g., "robomimic/lift/ph/image.hdf5")
    )
    dataset_path: str | None = (
        None  # Local path (deprecated, use dataset_repo/dataset_filename)
    )
    dataset_paths: list[str] | None = (
        None  # Multiple HDF5 paths [expert, rollout1, ...]
    )
    # Rollout HDF5 paths to mix in alongside the primary (expert) dataset.
    # Used by PushT, where the expert demos come from a zarr (dataset_repo /
    # dataset_filename / dataset_path) but collected rollouts are robomimic-
    # format HDF5 written by RolloutRecorder. Each path is tagged optimality=1
    # (null/play); the expert is optimality=0. None disables mixing (expert
    # only). On robomimic, mixing is instead done via dataset_paths above.
    rollout_dataset_paths: list[str] | None = None
    # Fraction of each rollout dataset actually USED (PushT mixing). 1.0 = all
    # demos (default). 0.5 = first half (collection order), etc. NOTE the
    # semantics are the INVERSE of val_dataset_percentage: this is the fraction
    # KEPT, not held out. Applied per rollout path, taking the first
    # round(n * fraction) demos. Lets you sweep the play:expert ratio without
    # pre-slicing HDF5 files. No-op on robomimic (which mixes via dataset_paths).
    rollout_use_fraction: float = 1.0
    filter_success: bool = (
        False  # Filter secondary datasets to keep only successful demos (reward > 0)
    )
    max_episode_steps: int = 400
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
    )
    obs_dim: int = -1
    act_dim: int = 10
    obs_steps: int = 2
    act_steps: int = 8
    horizon: int = 10  # Prediction horizon (typically obs_steps + act_steps)
    num_envs: int = 1
    save_video: bool = False
    shape_meta: dict = field(default_factory=dict)
    render_obs_key: str = "agentview_image"
    val_dataset_percentage: float = 0.0
    # Image observation settings
    rgb_model: str = "resnet18"
    resize_shape: list[int] | None = None
    crop_shape: list[int] | None = None
    random_crop: bool = True
    use_group_norm: bool = True
    use_seq: bool = True
    # REPA specific
    latent_type: str | None = None  # "lam" or "dino"
    use_precomputed: bool = (
        False  # True = load from HDF5, False = on-the-fly LAM inference
    )
    camera_keys: list[str] | None = (
        None  # e.g., ["agentview_image"]; None = auto-detect from HDF5
    )

    lam_frame_skips: list[int] | None = None  # e.g., [1, 8]; None = no LAM
    lam_latent_type: str | None = None  # None = no LAM; "prebn" or "bn" = enable

    dino_types: list[str] | None = None  # ["cls", "patch_mean"]
    dino_model: str | None = None  # e.g. vits16plus, vitb16
    dino_align_target: str | None = None  # "fd" or "id"
    dino_ckpt_dir: str = "/oscar/data/csun45/zzeng28/cache/torch/dinov3"
    dino_repo: str = "/oscar/data/csun45/zzeng28/cache/torch/dinov3/dinov3"


@dataclass
class LAMConfig:
    lam_image_channels: int = 3
    lam_model_dim: int = 1024
    lam_latent_dim: int = 32
    lam_patch_size: int = 16
    lam_enc_blocks: int = 16
    lam_dec_blocks: int = 16
    lam_num_heads: int = 16
    lam_dropout: float = 0.0
    lam_ckpt_path: str = None


@dataclass
class Config:
    optimization: OptimizationConfig
    network: NetworkConfig
    task: TaskConfig
    log: LogConfig
    lam: LAMConfig = None
    mode: str = "train"  # "train" or "eval"
