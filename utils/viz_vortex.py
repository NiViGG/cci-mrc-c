from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def plot_phase_space(hidden_states, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    h = torch.cat(hidden_states, dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    h2 = pca.fit_transform(h)
    plt.figure(figsize=(6, 5))
    plt.scatter(h2[:, 0], h2[:, 1], s=4, alpha=0.75)
    plt.title("Phase-space projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_cci_and_delta(cci_series, cci_out: str, delta_out: str):
    Path(cci_out).parent.mkdir(parents=True, exist_ok=True)
    cci = np.asarray(cci_series, dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(cci)
    plt.title("CCI over time")
    plt.xlabel("Window step")
    plt.ylabel("Bits")
    plt.tight_layout()
    plt.savefig(cci_out, dpi=160)
    plt.close()

    if len(cci) > 1:
        delta = np.diff(cci)
        plt.figure(figsize=(7, 4))
        plt.plot(delta)
        plt.title("Delta CCI")
        plt.xlabel("Window step")
        plt.ylabel("Delta bits")
        plt.tight_layout()
        plt.savefig(delta_out, dpi=160)
        plt.close()
