"""Offline builder for the goal-retrieval index.

Loads the same frozen IDM that downstream agents use, iterates the IDM
dataset, and encodes every (obs, goal_obs) pair into a (key, value) pair:

  - key   = query embedding (either flatten(z_t) or the IDM's obs_summary).
  - value = z_goal (raw frozen-encoder output of the goal frame).

Saves the stacked tensors to ``optimization.retrieval_index_path`` along with
metadata. The ``RetrievalGoalAgent`` loads this file at agent init and queries
it with the same ``retrieval_query_space``.

Usage:
    python scripts/build_goal_retrieval_index.py \
        task=tool_hang_ph_image_gp \
        network=goal_predictor_retrieval \
        network.encoder_type=image \
        optimization.idm_checkpoint_path=<path/to/idm.pt> \
        optimization.retrieval_query_space=obs_summary \
        optimization.retrieval_index_path=<path/to/index.pt>
"""

from __future__ import annotations

import os

# Set MuJoCo rendering backend before any robomimic imports.
os.environ["MUJOCO_GL"] = "egl"

import hydra
import loguru
import torch
from tensordict import TensorDict
from tqdm import tqdm

from mip.config import Config
from mip.datasets.robomimic_dataset import make_idm_dataset
from mip.flow_map import FlowMap
from mip.network_utils import get_encoder, get_network
from mip.torch_utils import limit_threads, set_seed


def _load_frozen_idm(config: Config, device: str):
    """Mirror the agent's IDM-load path so we always index in the same space."""
    net = get_network(config.network, config.task)
    flow_map = FlowMap(net).to(device)
    encoder = get_encoder(config.network, config.task).to(device)

    idm_path = config.optimization.idm_checkpoint_path
    if idm_path is None:
        raise ValueError("idm_checkpoint_path must be set to build the retrieval index")
    loguru.logger.info(f"Loading pretrained IDM from {idm_path}")
    state_dict = torch.load(idm_path, map_location=device, weights_only=False)
    flow_map.load_state_dict(state_dict["flow_map"])

    encoder_sd = state_dict["encoder"]
    if "uncond_emb" in encoder_sd:
        enc_out_dim = encoder_sd["uncond_emb"].shape[0]
    else:
        enc_out_dim = config.network.encoder_out_dim or config.network.emb_dim

    has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
    if has_goal_dropout:
        from mip.encoders import GoalDropoutEncoder

        encoder = GoalDropoutEncoder(
            encoder, enc_out_dim, config.task.obs_steps
        ).to(device)
        loguru.logger.info("IDM checkpoint has GoalDropoutEncoder, wrapping encoder")
    encoder.load_state_dict(encoder_sd)

    encoder.requires_grad_(False)
    flow_map.requires_grad_(False)
    encoder.eval()  # deterministic center crop — matches deployment
    flow_map.eval()

    # Strip the GoalDropoutEncoder wrapper for direct calls (we want the raw encoder).
    from mip.encoders import GoalDropoutEncoder

    inner_encoder = encoder.encoder if isinstance(encoder, GoalDropoutEncoder) else encoder

    return inner_encoder, flow_map, enc_out_dim


def _compute_query(
    z_t: torch.Tensor,
    query_space: str,
    flow_map_net: torch.nn.Module,
) -> torch.Tensor:
    """z_t: (B, To, D) -> query (B, D_q)."""
    if query_space == "flatten_zt":
        return z_t.flatten(1)
    elif query_space == "obs_summary":
        if not hasattr(flow_map_net, "obs_summarizer"):
            raise ValueError(
                "query_space='obs_summary' requires a v2 IDM with an obs_summarizer "
                f"head; got {type(flow_map_net).__name__}."
            )
        # Match LBMDiTIDMv2._summarize: obs_summarizer(flatten(condition[:, :To_obs])).
        To_obs = getattr(flow_map_net, "To_obs", z_t.shape[1])
        return flow_map_net.obs_summarizer(z_t[:, :To_obs].flatten(1))
    else:
        raise ValueError(
            f"Unknown query_space {query_space!r}; expected 'flatten_zt' or 'obs_summary'."
        )


@hydra.main(version_base=None, config_path="../examples/configs/", config_name="main")
def main(config: Config):
    set_seed(config.optimization.seed)
    limit_threads(1)

    out_path = config.optimization.retrieval_index_path
    if out_path is None:
        raise ValueError(
            "optimization.retrieval_index_path must be set to a destination .pt path"
        )

    device = config.optimization.device
    query_space = config.optimization.retrieval_query_space

    # task.obs_dim is normally populated by env reset; for index build we don't
    # need a live env, so set it from network.emb_dim for image tasks (matches
    # what the training scripts do).
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim

    encoder, flow_map, enc_out_dim = _load_frozen_idm(config, device)

    # Try to load the IDM's matching normalizer so embeddings exactly match
    # what downstream agents will see.
    import pickle

    normalizer = None
    normalizer_path = os.path.join(
        os.path.dirname(config.optimization.idm_checkpoint_path), "normalizer.pkl"
    )
    if os.path.exists(normalizer_path):
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)
        loguru.logger.info(f"Loaded IDM normalizer from {normalizer_path}")
    else:
        loguru.logger.warning(
            f"IDM normalizer not found at {normalizer_path}; falling back to a "
            "newly computed normalizer (this is usually a mistake — keys must "
            "be in the same space as the agent's queries)."
        )

    dataset = make_idm_dataset(config.task, normalizer=normalizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4,
        shuffle=False,        # ordered — keys[i] corresponds to dataset sample i
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )

    keys_all = []
    values_all = []
    n_total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Indexing ({query_space})"):
            if config.task.obs_type != "image":
                raise NotImplementedError(
                    "Index builder currently only supports image observations."
                )

            obs_dict = {
                k: batch["obs"][k][:, : config.task.obs_steps].to(device)
                for k in batch["obs"]
            }
            goal_dict = {k: batch["goal_obs"][k].to(device) for k in batch["goal_obs"]}

            B = next(iter(obs_dict.values())).shape[0]
            obs = TensorDict(obs_dict, batch_size=B)
            goal_obs = TensorDict(goal_dict, batch_size=B)

            z_t = encoder(obs, None)             # (B, To, D)
            z_goal = encoder(goal_obs, None)     # (B, 1, D)

            q = _compute_query(z_t, query_space, flow_map.net)  # (B, D_q)
            v = z_goal.squeeze(1)                                 # (B, D)

            keys_all.append(q.cpu())
            values_all.append(v.cpu())
            n_total += B

    keys = torch.cat(keys_all, dim=0)
    values = torch.cat(values_all, dim=0)

    payload = {
        "keys": keys,                    # (N, D_q)
        "values": values,                # (N, D)  — goal embedding
        "query_space": query_space,
        "to_obs": int(config.task.obs_steps),
        "emb_dim": int(enc_out_dim),
        "n": int(n_total),
        "idm_checkpoint_path": config.optimization.idm_checkpoint_path,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)
    loguru.logger.info(
        f"Saved retrieval index to {out_path} | n={n_total} | "
        f"key_dim={keys.shape[1]} | value_dim={values.shape[1]} | "
        f"query_space={query_space}"
    )


if __name__ == "__main__":
    main()
