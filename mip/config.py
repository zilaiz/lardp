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
    goal_stats_path: str | None = None  # path to precomputed goal normalization stats (.pt)
    # IDM + FDM joint training (lbmidm_v2 / IDMFDMAgent)
    fdm_loss_scale: float = 0.0  # weight for forward-dynamics auxiliary loss (0 = disabled)
    # Dropout annealing for condistill extra_cond_encoder
    extra_cond_dropout_warmup_steps: int = 5000  # number of steps to keep dropout at 0
    extra_cond_dropout_rampup_steps: int = 10000  # number of steps to linearly ramp dropout from 0 to max
    # Diagnostic: measure how much extra_cond affects teacher representations
    diagnose_teacher_delta: bool = False


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
    goal_align_depth: int | list[int] | None = 3
    # Goal predictor DiT with DDT head (encoder-decoder split)
    goal_ddt_enc_depth: int = 8
    goal_ddt_dec_depth: int = 2
    goal_ddt_d_model_enc: int | None = None  # None = use emb_dim
    goal_ddt_d_model_dec: int | None = None  # None = same as d_model_enc
    goal_ddt_n_heads_enc: int = 8
    goal_ddt_n_heads_dec: int = 8
    goal_ddt_dropout: float = 0.0
    # LBMDiTIDMv2 (IDM with obs summarizer + FDM head)
    obs_summarizer_hidden: int | None = None  # None -> 2 * obs_dim
    action_proj_hidden: int | None = None  # None -> 2 * obs_dim
    fdm_hidden: int | None = None  # None -> 2 * obs_dim


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
