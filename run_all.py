"""
run_all.py
----------
Entry point for the CCI + MRC-C research repository.

Runs all four experiments in sequence, prints results to the console,
and optionally saves visualizations to results/.

Usage:
    python run_all.py
"""

import sys
import os

# Ensure the repo root is on the Python path so submodule imports work
sys.path.insert(0, os.path.dirname(__file__))

from experiments.exp_feedforward_vs_rnn import run as run_exp1
from experiments.exp_training_dynamics import run as run_exp2
from experiments.exp_noise_vs_structure import run as run_exp3
from experiments.exp_transformer_baseline import run as run_exp4
from utils.viz import plot_cci_comparison, plot_training_dynamics


def main() -> None:
    print("=" * 60)
    print("  CCI + MRC-C  —  Run All Experiments")
    print("=" * 60)

    # ── Experiment 1: Feedforward vs RNN ────────────────────────
    print("\n[ Experiment 1 ] Feedforward vs Recurrent")
    print("-" * 40)
    results1 = run_exp1()

    # ── Experiment 2: Training Dynamics ─────────────────────────
    print("\n[ Experiment 2 ] Training Dynamics")
    print("-" * 40)
    results2 = run_exp2()

    # ── Experiment 3: Noise vs Structure ────────────────────────
    print("\n[ Experiment 3 ] Noise vs Structure")
    print("-" * 40)
    results3 = run_exp3()

    # ── Experiment 4: Transformer Baseline ──────────────────────
    print("\n[ Experiment 4 ] Transformer Baseline")
    print("-" * 40)
    results4 = run_exp4()

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Exp 1  Feedforward CCI : {results1['feedforward_cci']:.6f}")
    print(f"  Exp 1  Recurrent   CCI : {results1['recurrent_cci']:.6f}")
    print(f"  Exp 2  Final epoch CCI : {results2[-1]['cci']:.6f}")
    print(f"  Exp 3  Noise   input CCI : {results3['cci_noise']:.6f}")
    print(f"  Exp 3  Struct. input CCI : {results3['cci_structured']:.6f}")
    print(f"  Exp 4  Transformer   CCI : {results4['transformer_cci']:.6f}")
    print(f"  Exp 4  RNN           CCI : {results4['rnn_cci']:.6f}")

    # ── Plots ────────────────────────────────────────────────────
    print("\n[ Plots ]")
    print("-" * 40)
    plot_cci_comparison(
        labels=["Feedforward", "Recurrent", "Transformer"],
        cci_values=[
            results1["feedforward_cci"],
            results1["recurrent_cci"],
            results4["transformer_cci"],
        ],
        title="CCI by Architecture (Exp 1 + 4)",
        filename="cci_comparison_arch.png",
    )

    epochs = [r["epoch"] for r in results2]
    cci_vals = [r["cci"] for r in results2]
    loss_vals = [r["loss"] for r in results2]
    plot_training_dynamics(epochs, cci_vals, loss_vals)

    plot_cci_comparison(
        labels=["Noise Input", "Structured Input"],
        cci_values=[results3["cci_noise"], results3["cci_structured"]],
        title="CCI: Noise vs Structured Input (Exp 3)",
        filename="cci_noise_vs_structure.png",
    )

    print("\nDone.  Results written to results/")


if __name__ == "__main__":
    main()
