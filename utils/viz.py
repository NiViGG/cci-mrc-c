"""
utils/viz.py
------------
Visualization utilities for CCI experiment results.

All functions save figures to the results/ directory and are silently
skipped when a display is unavailable (e.g. headless CI environments).
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to results/ and close it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def plot_cci_comparison(
    labels: list[str],
    cci_values: list[float],
    title: str = "CCI Comparison",
    filename: str = "cci_comparison.png",
) -> None:
    """
    Bar chart comparing CCI values across models or conditions.

    Parameters
    ----------
    labels     : list of str
    cci_values : list of float
    title      : str
    filename   : str  (saved under results/)
    """
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.4), 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, cci_values, color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("CCI (nats)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, cci_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cci_values) * 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    _save(fig, filename)


def plot_training_dynamics(
    epochs: list[int],
    cci_values: list[float],
    loss_values: list[float],
    filename: str = "training_dynamics.png",
) -> None:
    """
    Dual-axis plot of CCI and training loss over epochs.

    Parameters
    ----------
    epochs      : list of int
    cci_values  : list of float
    loss_values : list of float
    filename    : str
    """
    fig, ax1 = plt.subplots(figsize=(7, 4))

    color_cci = "#4C72B0"
    color_loss = "#DD8452"

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CCI (nats)", color=color_cci, fontsize=11)
    ax1.plot(epochs, cci_values, color=color_cci, marker="o", label="CCI")
    ax1.tick_params(axis="y", labelcolor=color_cci)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Loss (MSE)", color=color_loss, fontsize=11)
    ax2.plot(epochs, loss_values, color=color_loss, marker="s", linestyle="--", label="Loss")
    ax2.tick_params(axis="y", labelcolor=color_loss)

    fig.suptitle("CCI Dynamics During Training", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)
