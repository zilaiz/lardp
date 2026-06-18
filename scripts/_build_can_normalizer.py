"""Rebuild the obs/action normalizer the can_ph_image joint_pt runs used.

The runs resolved task.dataset_paths -> data/robomimic/can/ph/image_v15.hdf5
(delta actions, act_dim=7, abs_action=False) via launch-time overrides, and
joint_pt computes a *fresh* normalizer from that dataset (idm_checkpoint_path
was None). The committed can_ph_image.yaml now points at image_v15_abs.hdf5,
so we override the task config back to what the run actually used and dump the
normalizer the viz script needs.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from hydra import initialize_config_dir

from mip.datasets.robomimic_dataset import make_idm_dataset

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "viz" / "can_joint_pt_compare" / "normalizer.pkl"


def main():
    config_abs = str((REPO / "examples/configs").resolve())
    overrides = [
        "task=can_ph_image",
        # Match the run's resolved task config (see wandb debug.log).
        "task.dataset_paths=[data/robomimic/can/ph/image_v15.hdf5]",
        "task.abs_action=false",
        "task.act_dim=7",
        "task.shape_meta.action.shape=[7]",
    ]
    with initialize_config_dir(config_dir=config_abs, version_base=None):
        cfg = hydra.compose(config_name="main", overrides=overrides)

    ds = make_idm_dataset(cfg.task, mode="train")
    norm = ds.normalizer
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(norm, f)

    print("obs keys:", list(norm["obs"].keys()))
    for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
        o = norm["obs"][k]
        print(f"  {k}: {type(o).__name__}  min={getattr(o,'min',None)} max={getattr(o,'max',None)}")
    print("action:", type(norm["action"]).__name__)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
