# Closed Causal Information (CCI) with MRC-C

Computable metric for autonomous internal dynamics in neural systems (CCI + MRC-C)

## Overview

CCI (Closed Causal Information) is a computable metric for measuring autonomous internal
dynamics in neural systems. It quantifies the degree to which a system's hidden state
evolves according to its own internal structure, independent of external input.

MRC-C (Minimal Recurrent Core — Causal) is the corresponding architectural analysis
framework that identifies the minimal recurrent subgraph responsible for generating
closed causal dynamics.

## Core Idea

```
E(X) = Σ I(H_t ; H_{t+1} | E_t)
```

This measures the mutual information between consecutive hidden states, conditioned on
external input `E_t`. High CCI indicates that future states are largely predictable from
current internal states, independent of incoming stimuli — a hallmark of autonomous
internal dynamics.

## What This Repo Shows

- Feedforward vs recurrent separation via CCI
- Emergence of internal dynamics under training
- Phase-space structure of recurrent attractors
- Transformer baseline comparison

## What This is NOT

- Not a theory of consciousness
- Not a claim about agency or sentience
- Not a philosophical argument — purely an empirical signal

## Repository Structure

```
cci-mrc-c/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── paper/
│   └── main.tex           # arXiv-ready paper draft
├── metrics/
│   └── cci.py             # CCI metric implementation
├── models/
│   └── mrc_c.py           # MRC-C model definitions
├── experiments/
│   ├── exp_feedforward_vs_rnn.py
│   ├── exp_training_dynamics.py
│   ├── exp_noise_vs_structure.py
│   └── exp_transformer_baseline.py
├── utils/
│   └── viz.py             # Visualization utilities
├── results/               # Output directory
├── docs/
│   └── notes.md
└── run_all.py             # Entry point
```

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py
```

## Outputs

- CCI values printed to console for each experiment
- Comparative tables (feedforward vs recurrent)
- Optional plots saved to `results/`

## Use Cases

- **Model diagnostics**: detect whether a model has meaningful internal temporal dynamics
- **Agent analysis**: quantify recurrence depth in RL or cognitive agents
- **Internal dynamics tracking**: monitor CCI across training to detect phase transitions

## Citation

If you use this work, please cite:

```bibtex
@misc{ccimrcc2024,
  title  = {Closed Causal Information (CCI) with MRC-C},
  year   = {2024},
  note   = {https://github.com/NiViGG/cci-mrc-c}
}
```

## License

MIT — see [LICENSE](LICENSE).
