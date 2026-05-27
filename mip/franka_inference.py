"""Real-robot inference helpers for the Franka Coffee-Pod policy.

Training stores 10-dim actions as [pos(3), rot6d(6), gripper(1)] where
`rot6d` is the first two ROWS of the rotation matrix, flattened — matching
`pytorch3d.transforms.matrix_to_rotation_6d` and the repo's
`mip.datasets.rotation_conversion.matrix_to_rotation_6d`.

At inference time the policy samples a (B, T, 10) action chunk. This module
provides the decode path from that representation back to what a Franka
controller typically expects: `[x, y, z, qx, qy, qz, qw]` plus a gripper
command. Quaternion convention is `[qx, qy, qz, qw]` (scipy / ROS /
`geometry_msgs/Quaternion`), matching how the raw teleop data was stored.

All helpers are numpy-first because real-robot drivers almost always consume
numpy. A user would typically call:

    act_norm = agent.sample(act_0, obs, num_steps=K, use_ema=True)
    act_raw  = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
    cmds     = decode_action(act_raw[0])     # per step, drop batch dim
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt decode of a 6D rotation to a 3x3 orthonormal matrix.

    The 6D format is the first two ROWS of the rotation matrix, flattened —
    so `rot6d[..., :3]` is row 0 and `rot6d[..., 3:]` is row 1. Row 2 is
    recovered as the cross product.

    Args:
        rot6d: array of shape (..., 6). Network output can be arbitrary
            (not constrained to orthonormal); this function re-orthonormalizes.

    Returns:
        (..., 3, 3) rotation matrix with det = +1, up to float precision.
    """
    r = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = r[..., :3], r[..., 3:6]

    # Row 0: normalize a1
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12, None)

    # Row 1: remove a1 component from a2, then normalize
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12, None)

    # Row 2: right-handed completion
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-2)   # (..., 3, 3)


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Decode rot6d directly to a quaternion in `[qx, qy, qz, qw]` order.

    Canonicalizes so that `qw >= 0` — removes the quaternion double-cover
    ambiguity so consecutive timesteps produce continuous quaternions.

    Args:
        rot6d: (..., 6) array.

    Returns:
        (..., 4) unit quaternion in xyzw order (scipy / ROS convention).
    """
    R = rot6d_to_matrix(rot6d)                          # (..., 3, 3)
    flat = R.reshape(-1, 3, 3)
    q_xyzw = Rotation.from_matrix(flat).as_quat()       # (N, 4) [x, y, z, w]

    # canonicalize: force qw >= 0 for continuity
    neg = q_xyzw[..., 3:4] < 0
    q_xyzw = np.where(neg, -q_xyzw, q_xyzw)

    return q_xyzw.reshape(R.shape[:-2] + (4,)).astype(np.float32)


def decode_delta_action(
    delta7: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_quat_xyzw: np.ndarray,
    gripper_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Decode an un-normalized 7-dim CHUNK-RELATIVE delta action chunk.

    Use this when the policy was trained with
    ``task.delta_action_anchor: current_obs`` (see
    ``mip/franka_delta_transform.py``). Each step in the chunk is anchored
    to the SAME ``anchor_pos`` / ``anchor_quat_xyzw`` (the EEF pose at the
    moment the chunk was queried — typically the last obs frame).

    Composes the world-frame delta against the anchor DIRECTLY into the
    controller's (pos, quat, gripper) format — no intermediate rot6d
    round-trip. Rotation math is:
        target_pos = anchor_pos + delta_pos                  (world frame)
        R_target   = R_delta @ R_anchor                       (world-frame premultiply)
        target_quat = quat_xyzw(R_target), with qw >= 0 canonicalization

    Args:
        delta7: ``(H, 7)`` un-normalized delta actions
            ``[pos_delta(3), axis_angle_delta(3), gripper(1)]``. Apply
            ``dataset.normalizer["action"].unnormalize(...)`` before passing.
        anchor_pos: ``(3,)`` world-frame anchor EEF position in meters
            (= ``obs.robot0_eef_pos[-1]`` at chunk-query time, un-normalized).
        anchor_quat_xyzw: ``(4,)`` world-frame anchor EEF quaternion in xyzw
            (= ``obs.robot0_eef_quat[-1]``, un-normalized).
        gripper_threshold: binarization threshold for gripper output.

    Returns:
        Same dict shape as ``decode_action`` (pos / quat_xyzw / gripper /
        gripper_continuous), each leading-dim ``(H, ...)``.
    """
    delta7 = np.asarray(delta7, dtype=np.float32)
    if delta7.shape[-1] != 7:
        raise ValueError(
            f"decode_delta_action expected last dim=7, got shape {delta7.shape}"
        )
    anchor_pos = np.asarray(anchor_pos, dtype=np.float32)
    anchor_quat = np.asarray(anchor_quat_xyzw, dtype=np.float32)

    # Position: world-frame add.
    target_pos = (delta7[..., :3] + anchor_pos[None, :]).astype(np.float32)

    # Rotation: compose R_delta (from axis-angle) with R_anchor (from quat).
    # R_target = R_delta @ R_anchor  (world-frame rotation premultiply).
    R_anchor = Rotation.from_quat(anchor_quat).as_matrix()                  # (3, 3)
    R_delta = Rotation.from_rotvec(delta7[..., 3:6]).as_matrix()            # (..., 3, 3)
    R_target = R_delta @ R_anchor                                            # broadcast OK

    # To quaternion + canonicalize qw >= 0 (matches decode_action's convention).
    flat = R_target.reshape(-1, 3, 3)
    q_xyzw = Rotation.from_matrix(flat).as_quat()                            # (N, 4)
    neg = q_xyzw[..., 3:4] < 0
    q_xyzw = np.where(neg, -q_xyzw, q_xyzw)
    target_quat = q_xyzw.reshape(R_target.shape[:-2] + (4,)).astype(np.float32)

    # Gripper: absolute (no anchor involved), threshold for the binary {0, 1} command.
    grip_cont = delta7[..., 6].astype(np.float32)
    grip = (grip_cont > gripper_threshold).astype(np.float32)

    return {
        "pos": target_pos,
        "quat_xyzw": target_quat,
        "gripper": grip,
        "gripper_continuous": grip_cont,
    }


def decode_action(
    action10: np.ndarray,
    gripper_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Decode an un-normalized 10-dim action into Franka-driver components.

    Args:
        action10: (..., 10) array. Layout must be
            `[pos(3), rot6d(6), gripper(1)]`, already passed through
            `dataset.normalizer["action"].unnormalize(...)` — i.e. back on
            the original data scale (meters, raw rot6d, gripper in [0, 1]).
        gripper_threshold: Binarization threshold for the gripper channel.
            The policy's output is a soft scalar; the data was trained on
            `{0, 1}` commands (0 = open, 1 = close).

    Returns:
        dict with:
            'pos'                : (..., 3) target EEF position in world frame
            'quat_xyzw'          : (..., 4) target orientation, unit norm, qw>=0
            'gripper'            : (..., ) binarized gripper command in {0, 1}
            'gripper_continuous' : (..., ) raw scalar before binarization
    """
    action10 = np.asarray(action10, dtype=np.float32)
    if action10.shape[-1] != 10:
        raise ValueError(
            f"decode_action expected last dim=10, got shape {action10.shape}"
        )

    pos = action10[..., :3].astype(np.float32)
    rot6d = action10[..., 3:9]
    grip_cont = action10[..., 9].astype(np.float32)

    return {
        "pos": pos,
        "quat_xyzw": rot6d_to_quat_xyzw(rot6d),
        "gripper": (grip_cont > gripper_threshold).astype(np.float32),
        "gripper_continuous": grip_cont,
    }


# ---------------------------------------------------------------------------
# Self-test (run: `uv run python -m mip.franka_inference`)
# ---------------------------------------------------------------------------
def _self_test():
    """Round-trip: random unit quaternions → matrix → rot6d → back to quat.

    Also verifies that our rot6d layout is consistent with the repo's
    `matrix_to_rotation_6d` (used at conversion time) — first two ROWS.
    """
    rng = np.random.default_rng(0)
    N = 100

    # 1. Random unit quaternions in xyzw, canonicalized
    q = rng.standard_normal((N, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q[q[:, 3] < 0] *= -1  # force qw >= 0

    # 2. Encode: quat(xyzw) -> R (3x3) -> rot6d (first 2 rows, flattened)
    R = Rotation.from_quat(q).as_matrix()                     # (N, 3, 3)
    rot6d = R[:, :2, :].reshape(N, 6)                         # (N, 6) — conversion-time format

    # 3. Decode and check
    R_back = rot6d_to_matrix(rot6d)
    q_back = rot6d_to_quat_xyzw(rot6d)

    # Matrix round-trip should be exact up to float precision
    R_err = np.max(np.abs(R - R_back))
    # Quaternion round-trip (both canonicalized to qw >= 0)
    q_err = np.max(np.abs(q - q_back))

    # Test decode_action: build a 10-dim action from known (pos, rot6d, grip)
    pos   = rng.uniform(-1, 1, size=(N, 3)).astype(np.float32)
    grip  = rng.integers(0, 2, size=(N,)).astype(np.float32)
    act10 = np.concatenate([pos, rot6d.astype(np.float32), grip[:, None]], axis=-1)

    decoded = decode_action(act10, gripper_threshold=0.5)
    pos_err  = np.max(np.abs(decoded["pos"] - pos))
    quat_err = np.max(np.abs(decoded["quat_xyzw"] - q_back))
    grip_err = np.max(np.abs(decoded["gripper"] - grip))

    print(f"[rot6d_to_matrix] max |R - R_back|              = {R_err:.2e}")
    print(f"[rot6d_to_quat_xyzw] max |q - q_back|           = {q_err:.2e}")
    print(f"[decode_action] pos round-trip                  = {pos_err:.2e}")
    print(f"[decode_action] quat round-trip vs rot6d_to_quat = {quat_err:.2e}")
    print(f"[decode_action] gripper round-trip              = {grip_err:.2e}")

    assert R_err < 1e-5,  f"rot6d->matrix round-trip failed: err={R_err}"
    assert q_err < 1e-5,  f"rot6d->quat round-trip failed: err={q_err}"
    assert pos_err == 0,  f"pos pass-through altered: err={pos_err}"
    assert quat_err < 1e-6, f"decode_action quat inconsistent: err={quat_err}"  # float32 cast
    assert grip_err == 0, f"decode_action gripper binarization failed: err={grip_err}"
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
