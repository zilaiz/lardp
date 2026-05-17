"""Export the data normalizer used by a Franka joint-DDT-frozen-DP training run.

The DP-frozen training pipeline (``train_franka_lbmdit_joint_ddt_frozen_dp.py``)
computes a fresh ``MinMaxNormalizer`` from the task's HDF5(s) at the start of
training and never writes it out — LBMDiT's source checkpoint doesn't carry
one, so there is no sibling ``normalizer.pkl`` to grab. For deployment via
``interface_lbmdit_joint_ddt_frozen_dp.py`` we need the *same* normalizer the
trainer saw (single HDF5, train split, val_pct from the task config).

This script rebuilds it from the saved hydra config: load the task, call
``make_idm_dataset(task_config, mode="train")``, grab ``.normalizer``,
pickle it. Output matches what the interface's ``_load_normalizer_from_pickle``
expects: ``{"obs": {key: Normalizer}, "action": Normalizer}``.

Usage:
    python scripts/export_franka_joint_normalizer.py \\
        --config_path outputs/2026-05-11/18-11-49/.hydra/config.yaml \\
        --out         checkpoints/franka_coffee_pod_cog_joint_dp_normalizer.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import torch
from omegaconf import OmegaConf

from mip.datasets.robomimic_dataset import make_idm_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the joint run's .hydra/config.yaml")
    parser.add_argument("--out", type=str, required=True,
                        help="Where to write the .pkl")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    dataset = make_idm_dataset(cfg.task, mode="train")
    base_dataset = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )
    normalizer = base_dataset.normalizer

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(normalizer, f)

    print(f"Wrote normalizer to {out_path}")
    print(f"  obs keys: {sorted(normalizer['obs'].keys())}")
    print(f"  action min={normalizer['action'].min}")
    print(f"  action max={normalizer['action'].max}")


if __name__ == "__main__":
    main()
