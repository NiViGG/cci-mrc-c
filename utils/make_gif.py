from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def make_gif(frames, out_path: str, fps: int = 5):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(out_path, images, fps=fps)


def make_attractor_gif(points_2d: np.ndarray, out_path: str, fps: int = 8, stride: int = 10) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(out_path).parent / "_tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    total = len(points_2d)
    for i in range(stride, total + 1, stride):
        frame_path = tmp_dir / f"frame_{i:05d}.png"
        plt.figure(figsize=(5, 5))
        plt.plot(points_2d[:i, 0], points_2d[:i, 1], linewidth=1.0, alpha=0.9)
        plt.scatter(points_2d[i - 1, 0], points_2d[i - 1, 1], s=20)
        plt.title("Latent attractor trajectory")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=120)
        plt.close()
        frame_paths.append(str(frame_path))

    make_gif(frame_paths, out_path=out_path, fps=fps)
