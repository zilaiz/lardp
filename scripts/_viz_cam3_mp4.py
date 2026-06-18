"""Render cam3 RGB frames of each rollout episode to an mp4 at 3x play speed.

Native capture cadence is ~15 fps (median frame dt ~0.0668s), so 3x speedup -> 45 fps.
Frames are written at a fixed output fps (each frame = equal duration), which is the
standard way to visualize these rollouts despite occasional capture gaps.
"""
import os
import imageio.v2 as imageio
from PIL import Image

SRC = "/users/zzeng28/data/zzeng28/datasets/joint_pt_os1_h24_rollout"
OUT = "/oscar/data/csun45/zzeng28/repo/lardp/joint_pt_os1_h24_rollout_cam3_mp4"
NATIVE_FPS = 15.0
SPEEDUP = 3.0
OUT_FPS = NATIVE_FPS * SPEEDUP

os.makedirs(OUT, exist_ok=True)

episodes = sorted(d for d in os.listdir(SRC) if d.startswith("episode_"))
print(f"{len(episodes)} episodes -> {OUT} @ {OUT_FPS:g} fps")

for ep in episodes:
    rgb_dir = os.path.join(SRC, ep, "cam3", "rgb")
    frames = sorted(os.listdir(rgb_dir))
    out_path = os.path.join(OUT, f"{ep}.mp4")
    writer = imageio.get_writer(
        out_path, fps=OUT_FPS, codec="libx264",
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for f in frames:
        writer.append_data(imageio.imread(os.path.join(rgb_dir, f)))
    writer.close()
    print(f"  {ep}: {len(frames)} frames -> {out_path}")

print("done")
