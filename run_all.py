from pathlib import Path

from experiments.run_ff_vs_rnn import run as run_baseline
from experiments.training import run as run_training
from utils.viz_vortex import plot_cci_and_delta, plot_phase_space


def main():
    Path("assets").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    run_baseline()
    cci_series, history_h = run_training()
    plot_cci_and_delta(cci_series, "assets/cci_training.png", "assets/delta_cci.png")
    plot_phase_space(history_h, "assets/phase_space.png")
    print("Done. Artifacts written to assets/ and results/.")


if __name__ == "__main__":
    main()
