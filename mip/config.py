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
    rollout_noise_std: float = 0.0  # Gaussian noise std injected into actions during rollout collection
    max_rollout_demos: int = 0  # Max rollout episodes to save (0 = unlimited)


@dataclass
class OptimizationConfig:
    seed: int = 0
    loss_type: str = "flow"
    loss_scale: float = 100.0
    cls_loss_scale: float = 0.03
    repa_scale: float = 0.0 # REPA loss coefficient
    s_align_depth: int | list[int] = 2
    t_align_depth: int | list[int] = 2
    norm_type: str = "l2"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_steps: int = 1
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
    goal_dropout_prob: float = 0.0  # Probability of zeroing out goal frame during IDM training
    # IDM encoder local-linearity regularizer weight (0 disables)
    local_linearity_coef: float = 0.0
    # Goal predictor specific
    idm_checkpoint_path: str | None = None  # Path to pretrained IDM checkpoint (for goal predictor training)
    state_matching_weight: float = 0.0  # Lambda for state-matching loss ||g_hat - encoder(s_{t+k})||^2
    cfg_scale: float = 1.0  # CFG scale for goal predictor inference (1.0 = no guidance)
    # Goal predictor DiT specific
    goal_flow_num_steps: int = 5  # ODE steps for goal generation at inference
    goal_flow_loss_scale: float = 1.0  # weight for state flow loss
    action_reg_weight: float = 1.0  # weight for action regularization loss
    action_reg_t_weighting: str = "linear"  # per-sample weighting of action loss by t_flow: "none" | "linear"
    action_reg_num_steps: int = 1  # K Euler steps from x_t -> g_hat for action_reg_loss; 1 = original one-step shortcut
    goal_stats_path: str | None = None  # path to precomputed goal normalization stats (.pt)
    delta_stats_path: str | None = None  # path to precomputed delta (= z_goal - z_last_obs) stats (.pt) for the delta predictor
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
    retrieval_temperature: float = 1.0   # softmax temperature for "weighted"
    # If True, retrieve both obs_summary (= index keys) and goal embedding
    # (= index values) from the nearest training row, and inject the pair
    # directly into the IDM action trunk via ``forward_with_summary``,
    # bypassing the IDM's obs_summarizer. Forces top1 and requires
    # query_space="obs_summary" + a v2 IDM with ``forward_with_summary``.
    retrieval_use_retrieved_pair: bool = False
    # IDM + FDM joint training (lbmidm_v2 / IDMFDMAgent)
    fdm_loss_scale: float = 0.0  # weight for forward-dynamics auxiliary loss (0 = disabled)
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
    extra_cond_dropout_rampup_steps: int = 10000  # number of steps to linearly ramp dropout from 0 to max
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
    # E2E variant only (LBMDiTJointE2EAgent): whether the target LayerNorm
    # has learnable gamma/beta. Default False removes the gamma->0 collapse
    # mode; flip to True for the UNITE-faithful variant (encoder gets more
    # expressive target normalization at the cost of a real collapse path).
    joint_target_ln_affine: bool = False
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
    # Mixed-data sampling (joint pipeline, expert + IDM rollouts):
    # Target fraction of each training batch drawn from the expert (primary)
    # dataset; rollouts get (1 - fraction). Implemented via WeightedRandomSampler
    # over the ConcatDataset, so the natural size imbalance between expert and
    # rollouts doesn't dilute expert-conditional updates. None disables the
    # reweighting (falls back to uniform shuffle = natural proportions). No-op
    # when the dataset is a single source (no rollouts).
    expert_sample_fraction: float | None = 0.5
    # DDT/decoupled-time variant only (LBMDiTJointDDTAgent):
    # If True, sample independent t_state and t_action per batch during
    # training (DF-style decoupled noising). If False, the same scalar t
    # is used for both streams (diagonal training, matches LBMDiTJointE2E).
    joint_decouple_t: bool = True
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
    joint_t_dist: str = "uniform"  # "uniform" | "logit_normal"
    joint_t_dist_mu: float = 0.0
    joint_t_dist_sigma: float = 1.0


@dataclass
class NetworkConfig:
    network_type: str = "mlp"  # "mlp" or "cnn"
    num_layers: int = 4
    emb_dim: int = 512
    dropout: float = 0.1
    encoder_dropout: float = 0.0
    encoder_type: str | None = None  # "mlp", "per_step_mlp", "identity", "image", "dino"
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
    dataset_paths: list[str] | None = None  # Multiple HDF5 paths [expert, rollout1, ...]
    filter_success: bool = False  # Filter secondary datasets to keep only successful demos (reward > 0)
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
    use_precomputed: bool = False  # True = load from HDF5, False = on-the-fly LAM inference
    camera_keys: list[str] | None = None  # e.g., ["agentview_image"]; None = auto-detect from HDF5

    lam_frame_skips: list[int] | None = None  # e.g., [1, 8]; None = no LAM
    lam_latent_type: str | None = None  # None = no LAM; "prebn" or "bn" = enable

    dino_types: list[str] | None = None  # ["cls", "patch_mean"]
    dino_model: str | None = None  # e.g. vits16plus, vitb16
    dino_align_target: str | None = None # "fd" or "id"
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
