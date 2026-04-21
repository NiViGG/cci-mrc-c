# Methods Note

## Estimator definition

Primary CCI quantity:

$$
I(H_t; H_{t+1}\mid E_t)
$$

The implementation uses a Gaussian conditional mutual information estimator via
covariance factorization and Cholesky log-determinants.

## Bias-correction method (permutation null)

Comparative reporting uses:

$$
\text{bias\_corrected}=\max(\text{raw\_cmi}-\mu_{\text{perm\_null}}, 0)
$$

where $\mu_{\text{perm\_null}}$ is estimated from shuffled $H_{t+1}$ runs.

## Assumptions

- Gaussian approximation for entropy/CMI terms.
- Sufficient sample count for stable covariance estimates.
- Correct conditioning variable choice (`E_t`) for task context.
- Consistent seed control for reproducibility checks.

## Known failure modes

- High-dimensional finite-sample inflation in raw CMI.
- Hidden-variable leakage that can overstate internal closure.
- Near-singular covariance requiring adaptive jitter.
- Architecture-specific variance effects that can invert raw-only comparisons.

## Interpretation guidance

- Use `bias_corrected` as primary cross-architecture comparator.
- Treat raw values as diagnostics, not standalone evidence.
- Treat `1.0 bit` as operational in this setup only.
- Apply non-claim policy from `CLAIMS.md` to all reporting.
