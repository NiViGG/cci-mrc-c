CCI is a computable diagnostic for internal temporal dependence in dynamical systems.

Core objective:
- separate input-driven transitions from internally sustained state evolution.
- keep metric claims strictly empirical and reproducible.

System model:
- H_{t+1} = F(H_t, E_t, epsilon_t)

Core quantity:
- E(X) = sum_t I(H_t ; H_{t+1} | E_t)

Interpretation:
- low CCI: mostly reactive/input-driven mapping
- high CCI: stronger internal temporal closure (within setup assumptions)
- 1.0 bit is an operational threshold in this setup, not a universal bound.
- CCI supports diagnostic comparison; it is not causal proof and not a consciousness claim.
