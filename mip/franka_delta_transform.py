"""Chunk-relative delta-action transform for the franka pipeline.

Matches the convention used by openpi / pi 0.5 / DROID / robosuite OSC_POSE:
  - All actions in a chunk are deltas relative to the SAME anchor (the
    chunk-start state, = last obs frame for the franka dataset).
  - Position deltas: WORLD frame, simple subtraction.
  - Rotation deltas: WORLD frame, 3-dim AXIS-ANGLE (not rot6d).
  - Gripper is the explicit exception — always absolute, never delta'd.

Why axis-angle for deltas (not rot6d): every established real-robot
codebase uses a 3-dim rotation rep at the deployment action interface
(openpi LIBERO adapter, robosuite OSC_POSE, Diffusion Policy on UR5, DROID,
OpenVLA, Octo). rot6d's only advantage — continuity over all of SO(3) — is
irrelevant for bounded-magnitude delta rotations. 3-dim is simpler, matches
controller input format, and aligns the franka pipeline with these refs.

Action layout (CHANGES from the absolute pipeline's 10-dim):
  abs (existing, in HDF5): [pos(3), rot6d(6), gripper(1)]   = 10-dim
  delta (new, model space): [pos_delta(3), axis_angle_delta(3), gripper(1)] = 7-dim

Rotation convention (world frame, matches robosuite OSC_POSE with
``control_delta=True``):
  R_delta_world[t] = R_action[t] @ R_anchor.T
  axis_angle_delta[t] = matrix_to_axis_angle(R_delta_world[t])
At deployment:
  R_action[t] = axis_angle_to_matrix(axis_angle_delta[t]) @ R_anchor

The forward (to_delta) and inverse (from_delta) round-trip exactly under
float32 — verified by the smoke test at the bottom of this file.

Author: Zilai Zeng
"""
from __future__ import annotations

import numpy as np
import torch

from mip.datasets.rotation_conversion import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


# ----------------------------- helpers -----------------------------------


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """xyzw -> 3x3 rotation matrix. mip.rotation_conversion uses wxyz convention."""
    q_wxyz = np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)
    return quaternion_to_matrix(torch.from_numpy(q_wxyz)).numpy()


def _rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    return rotation_6d_to_matrix(torch.from_numpy(d6)).numpy()


def _matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    return matrix_to_rotation_6d(torch.from_numpy(R)).numpy()


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(torch.from_numpy(R)).numpy()


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    return axis_angle_to_matrix(torch.from_numpy(aa)).numpy()


# ----------------------------- transforms --------------------------------


def to_delta(
    abs_actions: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Convert absolute 10-dim action chunk to chunk-relative 7-dim deltas.

    Args:
        abs_actions:      (H, 10) absolute actions [pos(3), rot6d(6), grip(1)].
        anchor_pos:       (3,) world-frame anchor position (= last obs eef_pos).
        anchor_quat_xyzw: (4,) world-frame anchor quaternion in xyzw order
                          (= last obs eef_quat).

    Returns:
        (H, 7) delta actions: [pos_delta(3), axis_angle_delta(3), gripper(1)].
    """
    assert abs_actions.ndim == 2 and abs_actions.shape[-1] == 10
    H = abs_actions.shape[0]

    # Position delta (world frame, simple subtraction)
    pos_abs = abs_actions[:, :3]                                   # (H, 3)
    pos_delta = pos_abs - anchor_pos[None, :]                      # (H, 3)

    # Rotation delta (world frame, R_delta = R_action @ R_anchor.T)
    R_anchor = _quat_xyzw_to_matrix(anchor_quat_xyzw)              # (3, 3)
    R_actions = _rot6d_to_matrix(abs_actions[:, 3:9])              # (H, 3, 3)
    R_delta = np.einsum("hij,kj->hik", R_actions, R_anchor)        # R @ R_anchor.T
    axis_angle_delta = _matrix_to_axis_angle(R_delta)              # (H, 3)

    # Gripper unchanged
    gripper = abs_actions[:, 9:10]                                 # (H, 1)

    out = np.concatenate(
        [pos_delta, axis_angle_delta, gripper], axis=-1,
    ).astype(abs_actions.dtype, copy=False)
    assert out.shape == (H, 7)
    return out


def from_delta(
    delta_actions: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Convert 7-dim chunk-relative deltas back to absolute 10-dim actions.

    Inverse of ``to_delta`` to within float32 round-trip precision.
    """
    assert delta_actions.ndim == 2 and delta_actions.shape[-1] == 7
    H = delta_actions.shape[0]

    # Position
    pos_abs = delta_actions[:, :3] + anchor_pos[None, :]           # (H, 3)

    # Rotation: R_action = R_delta @ R_anchor
    R_anchor = _quat_xyzw_to_matrix(anchor_quat_xyzw)              # (3, 3)
    R_delta = _axis_angle_to_matrix(delta_actions[:, 3:6])         # (H, 3, 3)
    R_actions = np.einsum("hij,jk->hik", R_delta, R_anchor)
    rot6d_abs = _matrix_to_rot6d(R_actions)                        # (H, 6)

    # Gripper unchanged
    gripper = delta_actions[:, 6:7]

    out = np.concatenate(
        [pos_abs, rot6d_abs, gripper], axis=-1,
    ).astype(delta_actions.dtype, copy=False)
    assert out.shape == (H, 10)
    return out


# ----------------------------- smoke test --------------------------------


def _round_trip_test():
    """Verify to_delta -> from_delta recovers absolute actions exactly."""
    rng = np.random.default_rng(0)
    H = 10

    from scipy.spatial.transform import Rotation as R

    # Random anchor pose
    anchor_pos = rng.normal(scale=0.5, size=3).astype(np.float32)
    anchor_R = R.random(random_state=rng).as_matrix().astype(np.float32)
    anchor_quat_xyzw = R.from_matrix(anchor_R).as_quat().astype(np.float32)

    # H absolute target poses + binary gripper
    abs_pos = anchor_pos[None, :] + rng.normal(
        scale=0.05, size=(H, 3),
    ).astype(np.float32)
    abs_R = np.stack([
        R.random(random_state=rng).as_matrix().astype(np.float32) for _ in range(H)
    ])
    abs_rot6d = _matrix_to_rot6d(abs_R)
    grip = rng.integers(0, 2, size=(H, 1)).astype(np.float32)
    abs_actions = np.concatenate([abs_pos, abs_rot6d, grip], axis=-1)  # (H, 10)

    # Forward
    delta = to_delta(abs_actions, anchor_pos, anchor_quat_xyzw)
    assert delta.shape == (H, 7), f"delta shape {delta.shape}"

    # Inverse
    recon = from_delta(delta, anchor_pos, anchor_quat_xyzw)
    assert recon.shape == (H, 10), f"recon shape {recon.shape}"

    # 1) Round-trip pos
    np.testing.assert_allclose(recon[:, :3], abs_actions[:, :3], atol=1e-5)
    # 2) Round-trip gripper
    np.testing.assert_allclose(recon[:, 9], abs_actions[:, 9], atol=1e-7)
    # 3) Round-trip rotation (via matrix to avoid rot6d sign ambiguity)
    R_recon = _rot6d_to_matrix(recon[:, 3:9])
    R_orig = _rot6d_to_matrix(abs_actions[:, 3:9])
    np.testing.assert_allclose(R_recon, R_orig, atol=1e-5)
    print(f"[ok] round-trip pos+rot+grip recovers original (H={H})")

    # 4) When abs == anchor pose: delta should be zero pos + zero axis-angle
    abs_at_anchor = np.concatenate([
        anchor_pos[None, :],
        _matrix_to_rot6d(anchor_R[None, :, :]),
        np.array([[1.0]], dtype=np.float32),
    ], axis=-1)
    delta_at_anchor = to_delta(abs_at_anchor, anchor_pos, anchor_quat_xyzw)
    np.testing.assert_allclose(delta_at_anchor[0, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(delta_at_anchor[0, 3:6], 0.0, atol=1e-6)
    np.testing.assert_allclose(delta_at_anchor[0, 6], 1.0, atol=1e-7)
    print("[ok] anchor-equals-action gives zero pos + zero axis-angle delta")

    # 5) Deployment-side reconstruction sanity
    current_pos = rng.normal(scale=0.5, size=3).astype(np.float32)
    current_R = R.random(random_state=rng).as_matrix().astype(np.float32)
    current_quat = R.from_matrix(current_R).as_quat().astype(np.float32)
    recon_at_current = from_delta(delta, current_pos, current_quat)
    np.testing.assert_allclose(
        recon_at_current[0, :3], current_pos + delta[0, :3], atol=1e-6,
    )
    # Rotation reconstruction: R_target = R_delta @ R_current
    R_delta_first = _axis_angle_to_matrix(delta[0:1, 3:6])[0]
    R_target_expected = R_delta_first @ current_R
    R_target_actual = _rot6d_to_matrix(recon_at_current[0:1, 3:9])[0]
    np.testing.assert_allclose(R_target_actual, R_target_expected, atol=1e-5)
    print("[ok] deployment-side reconstruction matches expected math "
          "(world-frame premultiply)")

    # 6) Delta-rotation magnitudes are bounded by typical inter-frame rotation
    # — should be small (a few degrees) for realistic teleop. Synthetic test
    # uses fully-random rotations so this just checks the math doesn't blow up.
    delta_rot_norm = np.linalg.norm(delta[:, 3:6], axis=-1)
    print(f"     (synthetic delta rotation magnitudes — radians): "
          f"min={delta_rot_norm.min():.3f}, max={delta_rot_norm.max():.3f}")

    print("\nALL ROUND-TRIP TESTS PASSED")


if __name__ == "__main__":
    _round_trip_test()
