#!/bin/bash
# Export per-dim goal z-score stats for the frozen-target PT ablation.
#
# Writes one stats file (``{"mean": (256,), "var": (256,)}``) per task from the
# FROZEN DP ``encoder_ema`` over that task's training goal observations — the
# fixed normalization the frozen target is z-scored by in
# LBMDiTJointPTFrozenTargetAgent (``goal_stats_path``). encoder_ema is the
# default (matches the agent's dp_use_encoder_ema=true).
#
# Two regimes:
#   expert (default): goals from the EXPERT dataset only  -> goal_stats/<task>_dp.pt
#                     (use with the *_idm / expert-only training runs, EO arm).
#   --mixed:          goals from the *_mixed config (EXPERT + ROLLOUT/play)
#                     -> goal_stats/<task>_dp_mixed.pt  (use with the EP arm so
#                     the normalization reflects the play-inclusive targets).
#
# Usage:
#   bash scripts/export_frozentgt_goal_stats.sh                 # expert, all five
#   bash scripts/export_frozentgt_goal_stats.sh can pusht       # expert, subset
#   bash scripts/export_frozentgt_goal_stats.sh --mixed         # mixed, all five
#   bash scripts/export_frozentgt_goal_stats.sh --mixed can     # mixed, subset
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p goal_stats

MODE="expert"
if [ "${1:-}" = "--mixed" ]; then MODE="mixed"; shift; fi

# Per-task frozen DP target encoders (canonical; same ckpts the sweeps bake in).
declare -A DP=(
  [can]="logs/can_ph_image_flow_None_lbmdit_256_seed1_horizon10_ed256_0.9/2026_06_04_00_24_15/models/model_best.pt"
  [square]="logs/square_ph_image_flow_None_lbmdit_256_seed1_horizon10_ed256_0.75/2026_06_04_00_24_15/models/model_best.pt"
  [transport]="logs/transport_ph_image_flow_None_lbmdit_256_seed1_horizon10_ed256_0.8/2026_06_04_00_24_14/models/model_best.pt"
  [tool_hang]="logs/tool_hang_ph_image_flow_None_lbmdit_256_seed2_horizon10_ed256_0.6/2026_06_04_00_24_13/models/model_best.pt"
  [pusht]="logs/pusht_ph_image_flow_None_lbmdit_256_seed0_horizon16_ed256_0.8/2026_06_13_22_03_30/models/model_best.pt"
)

tasks=("$@")
[ ${#tasks[@]} -eq 0 ] && tasks=(can square transport tool_hang pusht)

if [ "${MODE}" = "mixed" ]; then suffix="_dp_mixed"; else suffix="_dp"; fi

rc=0
for t in "${tasks[@]}"; do
  out="goal_stats/${t}${suffix}.pt"
  echo "=================== ${t} (${MODE}) -> ${out} ==================="
  if [ "${t}" = "pusht" ]; then
    tc="pusht_image"; [ "${MODE}" = "mixed" ] && tc="pusht_image_mixed"
    $PY scripts/compute_goal_stats_dp_pusht.py \
      --dp_checkpoint "${DP[$t]}" --output_path "${out}" \
      --config_dir examples/configs --task_config "${tc}" --network_config lbmdit || rc=$?
  elif [ "${MODE}" = "mixed" ]; then
    # _mixed config already lists [expert, rollout]; no --dataset_path override.
    $PY scripts/compute_goal_stats_dp.py \
      --dp_checkpoint "${DP[$t]}" --output_path "${out}" \
      --config_dir examples/configs --task_config "${t}_ph_image_mixed" --network_config lbmdit || rc=$?
  else
    $PY scripts/compute_goal_stats_dp.py \
      --dp_checkpoint "${DP[$t]}" --dataset_path "data/robomimic/${t}/ph/image_v15.hdf5" \
      --output_path "${out}" \
      --config_dir examples/configs --task_config "${t}_ph_image_idm" --network_config lbmdit || rc=$?
  fi
  echo "[${t}/${MODE}] exit=$? -> ${out}"
done
echo "DONE (rc=${rc})"
exit ${rc}
