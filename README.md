# CCI-MRC-C

Computable research pipeline for **Closed Causal Information (CCI)** in neural
dynamical systems.

Core metric:

\[
I(H_t ; H_{t+1}\mid E_t)
\]

This measures internal temporal dependence after conditioning on external input.

## Scientific framing (metric layer)

- We measure dependence, not will.
- CCI is a diagnostic metric, not causal proof.
- No consciousness claim is made from CCI alone.
- `1.0 bit` is an **operational threshold in our setup**, not a universal law.

## Vision framing (interpretive layer)

Interpretive hypotheses are allowed only as explicitly labeled interpretation
and must remain downstream of measured results and assumptions.

## Benchmarks included

- Feedforward baseline vs recurrent MRC-C.
- Noise vs structured input invariance check.
- Transformer temporal-dependence baseline.
- Training dynamics (`CCI`, `delta-CCI`, and jitter traces).

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

## Generated outputs

Results:
- `results/cci_values.json`
- `results/noise_vs_structure.json`
- `results/transformer_baseline.json`
- `results/training_dynamics.json`

Visuals:
- `assets/phase_space.png`
- `assets/cci_training.png`
- `assets/delta_cci.png`
- `assets/cci_comparison.png`
- `assets/attractor.gif`

## Current benchmark snapshot

From the latest seeded run:
- Feedforward baseline mean CCI: ~3.26 bits
- MRC-C mean CCI: ~24.97 bits
- Transformer baseline CCI: ~1.24 bits

These are setup-specific diagnostics and should be interpreted with
conditioning quality, estimator assumptions, and hidden-variable risk in mind.

## Scope and limitations

See `CLAIMS.md` and `theory/assumptions.md` for boundaries, assumptions, and
non-overclaim policy.

Practical rule: treat CCI as a **diagnostic dependence signal** inside the
modeled setup, not as causal proof and not as evidence of consciousness.
