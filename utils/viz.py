from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def plot_phase_space(hidden_states: list[torch.Tensor], out_path: str) -> np.ndarray:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    h = torch.cat(hidden_states, dim=0).detach().cpu().numpy()
    projection = PCA(n_components=2).fit_transform(h)
    plt.figure(figsize=(6, 5))
    plt.scatter(projection[:, 0], projection[:, 1], s=4, alpha=0.75)
    plt.title("Phase-space projection (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return projection


def plot_cci_series(cci_series: list[float], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cci_series, dtype=float)
    plt.figure(figsize=(7, 4))
    plt.plot(arr)
    plt.title("CCI over time")
    plt.xlabel("Window step")
    plt.ylabel("Bits")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_delta_cci(cci_series: list[float], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cci_series, dtype=float)
    delta = np.diff(arr) if len(arr) > 1 else np.asarray([0.0])
    plt.figure(figsize=(7, 4))
    plt.plot(delta)
    plt.title("Delta CCI")
    plt.xlabel("Window step")
    plt.ylabel("Delta bits")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


__all__ = ["plot_phase_space", "plot_cci_series", "plot_delta_cci"]
