import json
from pathlib import Path

import numpy as np

from utils.viz import plot_cci_series, plot_delta_cci


def run(results_dir: str = "results", assets_dir: str = "assets") -> None:
    Path(assets_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{results_dir}/cci_values.json", "r", encoding="utf-8") as f:
        compare = json.load(f)
    with open(f"{results_dir}/training_dynamics.json", "r", encoding="utf-8") as f:
        training = json.load(f)

    ff = [r["bias_corrected"] for r in compare["results"]["feedforward_baseline"]["per_run"]]
    rr = [r["bias_corrected"] for r in compare["results"]["mrc_c_core"]["per_run"]]
    cci_series = training["cci_series"]

    plot_cci_series(cci_series, f"{assets_dir}/cci_training.png")
    plot_delta_cci(cci_series, f"{assets_dir}/delta_cci.png")

    import matplotlib.pyplot as plt

    idx = np.arange(len(ff))
    width = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(idx - width / 2, ff, width=width, label="Feedforward")
    plt.bar(idx + width / 2, rr, width=width, label="MRC-C")
    plt.title("CCI Comparison Across Seeds")
    plt.xlabel("Run index")
    plt.ylabel("Bits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{assets_dir}/cci_comparison.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    run()
