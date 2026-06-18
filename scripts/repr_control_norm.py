"""Control: is EP's off-manifold edge the play DATA or just the merged NORMALIZER?

For each task, on the same rollout windows, compute the rollout-in-distribution
action probe R^2 for:
  EO @ N_exp   (what the main analysis used for EO)
  EO @ N_mix   (EO encoder, but fed the merged normalizer  <- the control)
  EP @ N_mix   (what the main analysis used for EP)

If EO@N_mix ~ EO@N_exp  and both << EP@N_mix, the EP advantage comes from the
encoder weights (play data), not the input normalizer.
"""
from __future__ import annotations

import numpy as np

import scripts.repr_extract as X
from scripts.repr_analyze import ridge_r2

TASKS = ["can", "square", "transport", "tool_hang"]


def rollout_action_r2(feat, act):
    nr = len(feat)
    rp = np.random.RandomState(1).permutation(nr)
    tr, va = rp[:int(0.7 * nr)], rp[int(0.7 * nr):]
    mu, sd = feat.mean(0, keepdims=True), feat.std(0, keepdims=True) + 1e-6
    f = (feat - mu) / sd
    y = act[:, 0, :]
    return ridge_r2(f[tr], y[tr], f[va], y[va])


def main():
    print(f"{'task':<11}{'EO@Nexp':>10}{'EO@Nmix':>10}{'EP@Nmix':>10}   verdict")
    for task in TASKS:
        cfg = X._cfg(task, "EP")
        sm = cfg.task.shape_meta.obs
        X.RGB = [k for k, v in sm.items() if v.get("type") == "rgb"]
        X.LOWDIM = [k for k, v in sm.items() if v.get("type", "low_dim") == "low_dim"]
        exp_path = cfg.task.dataset_paths[0]
        roll_path = cfg.task.dataset_paths[1]
        To, horizon = int(cfg.task.obs_steps), int(cfg.task.horizon)

        N_exp = X.fit_lowdim_normalizers(exp_path)
        N_mix = X.merge_lowdim(N_exp, X.fit_lowdim_normalizers(roll_path))
        roll = X.sample_windows(roll_path, To, horizon, X.CAP, seed=1)  # same seed as extract

        eo_enc, eo_ln = X.build_encoder(task, "EO")
        ep_enc, ep_ln = X.build_encoder(task, "EP")
        eo_nexp = X.encode(eo_enc, eo_ln, roll, N_exp, To)[0]
        eo_nmix = X.encode(eo_enc, eo_ln, roll, N_mix, To)[0]
        ep_nmix = X.encode(ep_enc, ep_ln, roll, N_mix, To)[0]

        r_eo_e = rollout_action_r2(eo_nexp, roll["act"])
        r_eo_m = rollout_action_r2(eo_nmix, roll["act"])
        r_ep_m = rollout_action_r2(ep_nmix, roll["act"])
        norm_effect = r_eo_m - r_eo_e
        data_effect = r_ep_m - r_eo_m
        verdict = ("DATA" if data_effect > 2 * abs(norm_effect) + 0.01 else "mixed")
        print(f"{task:<11}{r_eo_e:>10.3f}{r_eo_m:>10.3f}{r_ep_m:>10.3f}   "
              f"norm_eff={norm_effect:+.3f} data_eff={data_effect:+.3f} -> {verdict}")


if __name__ == "__main__":
    main()
