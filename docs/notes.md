# CCI Research Notes

## What is CCI?

**Closed Causal Information (CCI)** measures conditional temporal dependence
in the hidden-state trajectory of a neural system:

```
E(X) = (1 / (T-1)) * Σ_t  I(H_t ; H_{t+1} | E_t)
```

The key idea is to **separate internal dynamics from input-driven behaviour**.
Standard mutual information `I(H_t ; H_{t+1})` conflates the contribution of
the external input (which carries information about future inputs) with genuine
internal temporal structure.  By conditioning on `E_t` we remove the shared
information attributable to the input, leaving only the component that reflects
the system's own causal closure.

---

## Key Properties

| Property | Value |
|---|---|
| Feedforward network | CCI = 0 (exactly) |
| Untrained RNN | CCI ≈ small positive |
| Trained RNN (copying task) | CCI grows with training |
| Transformer (causal) | CCI > 0 due to attention over past |

---

## Estimation

We use a **Gaussian CMI estimator**:

```
I(A ; B | C) ≈ 0.5 * (log|Σ_AC| + log|Σ_BC| - log|Σ_C| - log|Σ_ABC|)
```

This is consistent for approximately Gaussian distributions and is
efficient to compute from sample covariance matrices.

---

## MRC-C

**Minimal Recurrent Core — Causal (MRC-C)** identifies the minimal subgraph
within a recurrent network that is responsible for the observed CCI.  It is
computed by greedy edge ablation: edges whose removal reduces CCI by more than
a threshold `ε` are considered causally load-bearing.

---

## Experiment Design

1. **Feedforward vs RNN**: confirms theoretical prediction (CCI = 0 vs CCI > 0)
2. **Training dynamics**: monitors CCI growth as a model learns a copying task
3. **Noise vs Structure**: confirms CCI is a model property, not input-dependent
4. **Transformer baseline**: quantifies how attention contributes to CCI

---

## Limitations

- The Gaussian estimator underestimates CCI for strongly non-Gaussian distributions
- Short trajectories (T < 100) yield high-variance estimates
- CCI does not distinguish the *content* of internal dynamics, only their magnitude
