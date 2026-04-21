import os
from pathlib import Path

from experiments.exp_ff_vs_rnn import run as run_ff_vs_rnn
from experiments.exp_noise_vs_structure import run as run_noise_vs_structure
from experiments.exp_transformer_baseline import run as run_transformer_baseline
from experiments.exp_training_dynamics import run as run_training_dynamics
from utils.make_gif import make_attractor_gif
from utils.make_plots import run as run_plot_generation
from utils.seed import set_seed
from utils.viz import plot_phase_space


def main():
    set_seed(42)
    Path("assets").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    smoke = os.getenv("CCI_SMOKE", "0") == "1"
    if smoke:
        run_ff_vs_rnn(seeds=[42], seq_len=128, n_perm=5)
        run_noise_vs_structure(seq_len=128, n_perm=5)
        run_transformer_baseline(seq_len=128, n_perm=5)
        print("Smoke run complete. Core results written to results/.")
        return

    run_ff_vs_rnn()
    run_noise_vs_structure()
    run_transformer_baseline()
    training_out = run_training_dynamics()
    projection = plot_phase_space(training_out["history_h"], "assets/phase_space.png")
    make_attractor_gif(projection, out_path="assets/attractor.gif")
    run_plot_generation()
    print("Done. Artifacts written to assets/ and results/.")


if __name__ == "__main__":
    main()
