"""Goal Predictor Retrieval Agent: non-parametric goal lookup.

Given the current obs encoding ``z_t``, retrieve the K nearest training
``z_t`` from a precomputed index and return the corresponding training goal
embedding(s) as ``g_hat``. The goal embedding is on-manifold by construction
(it's literally a real training embedding), so no FM, no L2, no mode collapse,
no off-manifold drift.

There is nothing to train. The "agent" is a thin wrapper around:
    1. The frozen IDM (encoder + flow_map) loaded from a checkpoint.
    2. A retrieval index loaded from disk (built by
       ``scripts/build_goal_retrieval_index.py``).

At ``sample()``:
    z_t  = encoder(obs)
    q    = compute_query(z_t)        # flatten or obs_summary
    idx  = topk(distance(q, keys))
    g_hat = aggregate(values[idx])
    return ode_sampler(... obs_emb=concat(z_t, g_hat) ...)
"""

from __future__ import annotations

from copy import deepcopy

import loguru
import numpy as np
import torch
import torch.nn as nn

from mip.config import Config
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.network_utils import get_encoder, get_network
from mip.samplers import ode_sampler
from mip.torch_utils import at_least_ndim


def _compute_query(
    z_t: torch.Tensor,
    query_space: str,
    flow_map_net: nn.Module,
) -> torch.Tensor:
    """z_t: (B, To, D) -> query (B, D_q). Mirrors the index builder."""
    if query_space == "flatten_zt":
        return z_t.flatten(1)
    elif query_space == "obs_summary":
        if not hasattr(flow_map_net, "obs_summarizer"):
            raise ValueError(
                "query_space='obs_summary' requires a v2 IDM with an obs_summarizer "
                f"head; got {type(flow_map_net).__name__}."
            )
        To_obs = getattr(flow_map_net, "To_obs", z_t.shape[1])
        return flow_map_net.obs_summarizer(z_t[:, :To_obs].flatten(1))
    else:
        raise ValueError(
            f"Unknown query_space {query_space!r}; expected 'flatten_zt' or 'obs_summary'."
        )


def _pairwise_distance(
    q: torch.Tensor,    # (B, D_q)
    k: torch.Tensor,    # (N, D_q)
    metric: str,
) -> torch.Tensor:       # (B, N)
    """Pairwise distance between query rows and key rows."""
    if metric == "l2":
        # ||q - k||^2 = ||q||^2 + ||k||^2 - 2 q.k  — keep squared (monotonic in distance).
        return torch.cdist(q, k, p=2)
    elif metric == "cosine":
        q_n = torch.nn.functional.normalize(q, dim=-1)
        k_n = torch.nn.functional.normalize(k, dim=-1)
        # Cosine distance in [0, 2]; smaller is closer.
        return 1.0 - q_n @ k_n.t()
    else:
        raise ValueError(f"Unknown distance metric {metric!r}; expected 'l2' or 'cosine'.")


class RetrievalGoalEncoderWrapper(BaseEncoder):
    """Wraps frozen encoder + retrieval index into an ``encoder``-shaped object
    that ``ode_sampler`` can call once per env step.

    Default mode (``use_retrieved_pair=False``):
        z_t   = encoder(obs)
        q     = compute_query(z_t)              # flatten or obs_summary
        idx   = topk(distance(q, keys))
        g_hat = aggregate(values[idx])
        return concat([z_t, g_hat])             # (B, To+1, D), IDM summarizes obs

    Snap-to-pair mode (``use_retrieved_pair=True``, requires query_space=
    "obs_summary"): the index's *keys* are themselves the training-time
    obs_summaries, so we retrieve both halves of the AdaLN conditioning at
    once and feed them to the IDM action trunk via ``forward_with_summary``,
    bypassing the IDM's obs_summarizer. The wrapper still has to encode the
    live obs to *compute the query*, but that encoding is not used as the
    IDM input.

    In snap-to-pair mode the wrapper returns ``(obs_summary, goal_emb)`` of
    shape ``(B, 2, D)`` instead of an obs/goal stack, and the agent routes
    sampling through a custom action-trunk ODE rather than ``ode_sampler``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        flow_map_net: nn.Module,
        keys: torch.Tensor,                 # (N, D_q)
        values: torch.Tensor,               # (N, D)
        query_space: str,
        distance: str,
        top_k: int,
        aggregation: str,
        temperature: float,
        use_retrieved_pair: bool,
    ):
        super().__init__()
        self.encoder = encoder
        # flow_map_net is needed for obs_summary query; held as a non-module
        # reference so its params don't end up in this wrapper's parameter list.
        object.__setattr__(self, "_flow_map_net", flow_map_net)
        self.register_buffer("keys", keys)
        self.register_buffer("values", values)
        self.query_space = query_space
        self.distance = distance
        self.top_k = top_k
        self.aggregation = aggregation
        self.temperature = temperature
        self.use_retrieved_pair = use_retrieved_pair

    def _retrieve(self, obs, mask):
        """Encode live obs, run kNN, return (idx, topk_dists)."""
        z_t = self.encoder(obs, mask)                               # (B, To, D)
        q = _compute_query(z_t, self.query_space, self._flow_map_net)
        d = _pairwise_distance(q, self.keys, self.distance)         # (B, N)
        topk = torch.topk(d, k=self.top_k, dim=-1, largest=False)
        return z_t, topk.indices, topk.values

    def retrieve_summary_pair(self, obs, mask=None):
        """Snap-to-pair mode: return (obs_summary, goal_emb) from row idx[:,0].

        Both vectors are real training-time embeddings; the live obs is only
        used to compute the query.
        """
        if self.query_space != "obs_summary":
            raise ValueError(
                "use_retrieved_pair=True requires query_space='obs_summary' so "
                "that index keys ARE training obs_summaries; got "
                f"{self.query_space!r}."
            )
        _, idx, _ = self._retrieve(obs, mask)
        top1 = idx[:, 0]                                            # (B,)
        obs_summary = self.keys[top1]                               # (B, D)
        goal_emb = self.values[top1]                                # (B, D)
        return obs_summary, goal_emb

    def forward(self, obs, mask=None):
        # Default (non-snap) path: ode_sampler-compatible (B, To+1, D) stack.
        z_t, idx, topk_dists = self._retrieve(obs, mask)
        candidates = self.values[idx]                               # (B, K, D)

        if self.aggregation == "top1" or self.top_k == 1:
            g_hat = candidates[:, 0:1]                              # (B, 1, D)
        elif self.aggregation == "weighted":
            w = torch.softmax(
                -topk_dists / max(self.temperature, 1e-6), dim=-1,
            )
            g_hat = (w.unsqueeze(-1) * candidates).sum(dim=1, keepdim=True)
        else:
            raise ValueError(
                f"Unknown aggregation {self.aggregation!r}; expected "
                "'top1' or 'weighted'."
            )

        return torch.cat([z_t, g_hat], dim=1)                       # (B, To+1, D)


class RetrievalGoalAgent:
    """Non-parametric retrieval-based goal predictor through a frozen IDM.

    No optimizer, no EMA, no training step. ``update`` raises; the agent is
    eval-only.
    """

    def __init__(self, config: Config):
        self.config = config
        device = config.optimization.device

        # --- Build IDM architecture and load weights ---
        net = get_network(config.network, config.task)
        self.flow_map = FlowMap(net).to(device)
        self.encoder = get_encoder(config.network, config.task).to(device)

        idm_path = config.optimization.idm_checkpoint_path
        if idm_path is None:
            raise ValueError("idm_checkpoint_path must be set for RetrievalGoalAgent")
        loguru.logger.info(f"Loading pretrained IDM from {idm_path}")
        state_dict = torch.load(idm_path, map_location=device, weights_only=False)
        self.flow_map.load_state_dict(state_dict["flow_map"])

        encoder_sd = state_dict["encoder"]
        if "uncond_emb" in encoder_sd:
            enc_out_dim = encoder_sd["uncond_emb"].shape[0]
        else:
            enc_out_dim = config.network.encoder_out_dim or config.network.emb_dim

        has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
        if has_goal_dropout:
            from mip.encoders import GoalDropoutEncoder

            self.encoder = GoalDropoutEncoder(
                self.encoder, enc_out_dim, config.task.obs_steps
            ).to(device)
        self.encoder.load_state_dict(encoder_sd)

        self.encoder.requires_grad_(False)
        self.flow_map.requires_grad_(False)
        self.flow_map.eval()

        from mip.encoders import GoalDropoutEncoder

        if isinstance(self.encoder, GoalDropoutEncoder):
            self._inner_encoder = self.encoder.encoder
            self._uncond_emb = self.encoder.uncond_emb
        else:
            self._inner_encoder = self.encoder
            self._uncond_emb = None

        idm_net = self.flow_map.net
        if hasattr(idm_net, "To_obs") and config.task.obs_steps != idm_net.To_obs:
            raise ValueError(
                f"task.obs_steps ({config.task.obs_steps}) must match the IDM's "
                f"To_obs ({idm_net.To_obs})."
            )

        # --- Load retrieval index ---
        index_path = config.optimization.retrieval_index_path
        if index_path is None:
            raise ValueError(
                "optimization.retrieval_index_path must be set for RetrievalGoalAgent"
            )
        loguru.logger.info(f"Loading retrieval index from {index_path}")
        payload = torch.load(index_path, map_location=device, weights_only=False)

        keys = payload["keys"].to(device).contiguous()
        values = payload["values"].to(device).contiguous()
        index_query_space = payload["query_space"]

        cfg_query_space = config.optimization.retrieval_query_space
        if cfg_query_space != index_query_space:
            raise ValueError(
                f"Index was built with query_space={index_query_space!r} but config "
                f"requests {cfg_query_space!r}; rebuild the index or update config."
            )

        # Validate the snap-to-pair flag.
        use_retrieved_pair = config.optimization.retrieval_use_retrieved_pair
        if use_retrieved_pair:
            if cfg_query_space != "obs_summary":
                raise ValueError(
                    "retrieval_use_retrieved_pair=True requires "
                    "retrieval_query_space='obs_summary' so the index's keys "
                    f"are themselves training obs_summaries; got {cfg_query_space!r}."
                )
            if not hasattr(idm_net, "forward_with_summary"):
                raise TypeError(
                    "retrieval_use_retrieved_pair=True requires an IDM with a "
                    f"forward_with_summary method (LBMDiTIDMv2); got "
                    f"{type(idm_net).__name__}."
                )
            if config.optimization.retrieval_top_k != 1 or (
                config.optimization.retrieval_aggregation != "top1"
            ):
                loguru.logger.warning(
                    "retrieval_use_retrieved_pair=True forces top1; ignoring "
                    f"top_k={config.optimization.retrieval_top_k}, "
                    f"aggregation={config.optimization.retrieval_aggregation!r}."
                )

        loguru.logger.info(
            f"Index loaded: n={keys.shape[0]} | key_dim={keys.shape[1]} | "
            f"value_dim={values.shape[1]} | query_space={index_query_space} | "
            f"use_retrieved_pair={use_retrieved_pair}"
        )

        # --- Build the wrapper ---
        # When use_retrieved_pair is on, the wrapper internally forces top1
        # regardless of the configured top_k/aggregation values.
        self.wrapper_encoder = RetrievalGoalEncoderWrapper(
            encoder=self._inner_encoder,
            flow_map_net=idm_net,
            keys=keys,
            values=values,
            query_space=cfg_query_space,
            distance=config.optimization.retrieval_distance,
            top_k=(1 if use_retrieved_pair else config.optimization.retrieval_top_k),
            aggregation=(
                "top1" if use_retrieved_pair else config.optimization.retrieval_aggregation
            ),
            temperature=config.optimization.retrieval_temperature,
            use_retrieved_pair=use_retrieved_pair,
        ).to(device)
        self._use_retrieved_pair = use_retrieved_pair

        # The retrieval agent has no optimizer; expose a sentinel so the harness
        # doesn't crash if it tries to introspect one.
        self.optimizer = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def sample(
        self,
        act_0: torch.Tensor,
        obs,
        num_steps: int = -1,
        use_ema: bool = True,        # ignored — no EMA in retrieval
    ) -> torch.Tensor:
        """Encode obs, retrieve goal, run frozen IDM action sampler.

        Two paths:
          * ``use_retrieved_pair=True``: retrieve (obs_summary, goal_emb) from
            the index, inject directly into the IDM action trunk via
            ``forward_with_summary``, bypassing the IDM's obs_summarizer.
          * default: build (B, To+1, D) stack and run the standard
            ``ode_sampler`` (or CFG variant).
        """
        del use_ema
        if num_steps >= 1:
            config = deepcopy(self.config.optimization)
            config.num_steps = int(num_steps)
        else:
            config = self.config.optimization

        with torch.no_grad():
            if self._use_retrieved_pair:
                act = self._snap_pair_sample(config, act_0, obs)
            elif config.cfg_scale != 1.0 and self._uncond_emb is not None:
                act = self._cfg_sample(config, act_0, obs)
            else:
                act = ode_sampler(
                    config, self.flow_map, self.wrapper_encoder, act_0, obs,
                )
        return act

    def _snap_pair_sample(self, config, act_0: torch.Tensor, obs) -> torch.Tensor:
        """Action ODE that injects retrieved (obs_summary, goal_emb) directly
        into the IDM action trunk, bypassing the obs_summarizer.

        Mirrors ``ode_sampler``'s loop but calls
        ``flow_map.net.forward_with_summary`` instead of
        ``flow_map.get_velocity`` so the conditioning never round-trips
        through ``_summarize``.
        """
        obs_summary, goal_emb = self.wrapper_encoder.retrieve_summary_pair(obs)
        # (B, D) each.

        device = act_0.device
        B = act_0.shape[0]
        num_steps = config.num_steps
        t_schedule = np.linspace(0, 1, num_steps + 1)
        if config.sample_mode == "stochastic":
            act_s = torch.randn_like(act_0, device=device)
        else:
            act_s = torch.zeros_like(act_0, device=device)

        for i in range(num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((B,), s_val, device=device)
            v, _ = self.flow_map.net.forward_with_summary(
                act_s, s, s, obs_summary, goal_emb,
            )
            act_s = act_s + v * (t_val - s_val)

        return act_s

    def _cfg_sample(self, config, act_0: torch.Tensor, obs) -> torch.Tensor:
        """CFG path: blend uncond and cond IDM velocities.

        Delegates retrieval to ``self.wrapper_encoder`` so all three modes
        (top1, weighted, use_retrieved_obs) behave consistently with the
        non-CFG path. The unconditional branch reuses whichever ``z_t`` the
        wrapper put in cond_emb (live or retrieved) so only the goal token
        differs between cond and uncond — which is what CFG actually toggles.
        """
        cond_emb = self.wrapper_encoder(obs, None)                  # (B, To+1, D)
        B = cond_emb.shape[0]
        device = cond_emb.device

        uncond = self._uncond_emb.expand(B, 1, -1)
        uncond_emb = torch.cat([cond_emb[:, :-1], uncond], dim=1)   # same z_t, uncond goal

        num_steps = config.num_steps
        t_schedule = np.linspace(0, 1, num_steps + 1)
        if config.sample_mode == "stochastic":
            act_s = torch.randn_like(act_0, device=device)
        else:
            act_s = torch.zeros_like(act_0, device=device)

        for i in range(num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((B,), s_val, device=device)
            t = torch.full((B,), t_val, device=device)

            v_uncond = self.flow_map.get_velocity(s, act_s, uncond_emb)
            v_cond = self.flow_map.get_velocity(s, act_s, cond_emb)
            v_cfg = v_uncond + config.cfg_scale * (v_cond - v_uncond)

            s_expanded = at_least_ndim(s, act_s.dim())
            t_expanded = at_least_ndim(t, act_s.dim())
            act_s = act_s + v_cfg * (t_expanded - s_expanded)

        return act_s

    # ------------------------------------------------------------------
    # No-ops to satisfy the harness API
    # ------------------------------------------------------------------
    def update(self, *args, **kwargs):
        raise RuntimeError(
            "RetrievalGoalAgent has no trainable parameters; call agent.sample() instead."
        )

    def save(self, path: str, training_state: dict = None):
        # Nothing to checkpoint, but keep a stub so the harness's save_agent
        # doesn't fail — emit just the metadata pointing at the index used.
        torch.save(
            {
                "idm_checkpoint_path": self.config.optimization.idm_checkpoint_path,
                "retrieval_index_path": self.config.optimization.retrieval_index_path,
                "training_state": training_state,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = False):
        # Index + IDM are loaded in __init__; nothing else to restore.
        del path, load_optimizer
        return None

    def eval(self):
        self.encoder.eval()

    def train(self):
        # Encoder train mode would re-enable random crop; for retrieval we want
        # the queries to match the keys (built in eval mode), so stay in eval.
        self.encoder.eval()
