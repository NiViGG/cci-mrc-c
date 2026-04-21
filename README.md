# CCI-MRC-C: Measuring Internal Temporal Dependence

## Overview

This repository implements Closed Causal Information (CCI), a computable metric
for internal temporal dependence in neural dynamical systems.

The core quantity is:

I(H_t ; H_{t+1} | E_t)

which measures how much future internal state remains predictable from current
internal state after conditioning on environment input.

## Why this matters

Standard output metrics can miss latent persistence and internal loops. CCI adds
a diagnostic layer for distinguishing reactive mappings from stateful dynamics.

## Operational threshold note

The 1.0 bit value is an observed operational threshold in this experimental
setup. It is not a universal constant and should be interpreted together with
model class, conditioning quality, and hidden-variable assumptions.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

## Output

- `results/cci_values.json`
- `assets/phase_space.png`
- `assets/cci_training.png`
- `assets/delta_cci.png`

## Scientific boundaries

CCI measures temporal dependence, not will or intent. See `CLAIMS.md` for scope,
risk disclosures, and interpretation rules.
