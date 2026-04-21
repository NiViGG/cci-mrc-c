from experiments.exp_ff_vs_rnn import run as run_experiment
from utils.seed import set_seed


def run() -> dict:
    set_seed(42)
    payload = run_experiment(seeds=[42, 43, 44, 45, 46], seq_len=700, env_dim=5)
    print("Feedforward mean:", payload["results"]["feedforward_baseline"]["mean"])
    print("Recurrent mean:", payload["results"]["mrc_c_core"]["mean"])
    return payload


if __name__ == "__main__":
    run()
