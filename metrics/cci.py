"""
metrics/cci.py
--------------
Closed Causal Information (CCI) estimator.

CCI measures the conditional mutual information between consecutive hidden
states given the external input:

    E(X) = (1 / (T-1)) * sum_t I(H_t ; H_{t+1} | E_t)

We use a Gaussian estimator based on regularised sample covariance matrices.
Tikhonov regularisation (adding eps * I to each covariance) ensures numerical
stability when inputs are near-singular (e.g. low-rank structured signals).
To avoid bias from very high-dimensional representations relative to the
number of samples, we first project hidden states onto their leading
principal components.
"""

import numpy as np

# Ridge regularisation added to every covariance matrix before log-det.
_REG_EPS: float = 1e-4

# Maximum number of PCA components to retain for hidden states.
# Keeps dimensionality well below the sample count.
_MAX_PCA_COMPONENTS: int = 32


def _regularised_cov(a: np.ndarray, eps: float = _REG_EPS) -> np.ndarray:
    """Return the regularised sample covariance matrix (rows = samples)."""
    c = np.cov(a, rowvar=False, ddof=1)
    c = np.atleast_2d(c)
    c += eps * np.eye(c.shape[0])
    return c


def _log_det(m: np.ndarray) -> float:
    """Numerically stable log-determinant of a positive-definite matrix."""
    sign, logdet = np.linalg.slogdet(m)
    if sign <= 0:
        return -np.inf
    return float(logdet)


def _pca_project(x: np.ndarray, n_components: int) -> np.ndarray:
    """
    Project x onto its leading `n_components` principal components.

    Parameters
    ----------
    x            : np.ndarray, shape (N, D)
    n_components : int  — must be < min(N, D)

    Returns
    -------
    np.ndarray, shape (N, n_components)
    """
    x_c = x - x.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(x_c, full_matrices=False)
    return x_c @ Vt[:n_components].T


def gaussian_cmi(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Estimate I(A ; B | C) under a multivariate Gaussian assumption.

    Uses regularised covariance matrices to handle near-singular inputs.

    Parameters
    ----------
    a, b, c : np.ndarray, shape (N, d_x)
        Column-wise feature matrices; rows are i.i.d. samples.

    Returns
    -------
    float
        Non-negative estimate of I(A ; B | C) in nats.
    """
    ac = np.concatenate([a, c], axis=1)
    bc = np.concatenate([b, c], axis=1)
    abc = np.concatenate([a, b, c], axis=1)

    # I(A;B|C) = 0.5 * (log|Σ_AC| + log|Σ_BC| - log|Σ_C| - log|Σ_ABC|)
    cmi = 0.5 * (
        _log_det(_regularised_cov(ac))
        + _log_det(_regularised_cov(bc))
        - _log_det(_regularised_cov(c))
        - _log_det(_regularised_cov(abc))
    )
    return float(max(cmi, 0.0))


def compute_cci(
    hidden_states: np.ndarray,
    external_inputs: np.ndarray,
    pca_components: int = _MAX_PCA_COMPONENTS,
) -> float:
    """
    Compute the Closed Causal Information (CCI) for a trajectory.

    Parameters
    ----------
    hidden_states  : np.ndarray, shape (T, d_h)
        Hidden-state trajectory.  Row t is H_t.
    external_inputs : np.ndarray, shape (T, d_e)
        External-input trajectory.  Row t is E_t.
    pca_components : int
        Number of PCA components to project hidden states onto before
        computing CCI.  Reduces estimation bias when d_h >> N.

    Returns
    -------
    float
        CCI estimate in nats.

    Raises
    ------
    ValueError
        If the trajectory is too short or the shapes are incompatible.
    """
    if hidden_states.ndim != 2 or external_inputs.ndim != 2:
        raise ValueError("hidden_states and external_inputs must be 2-D arrays.")
    T = hidden_states.shape[0]
    if T != external_inputs.shape[0]:
        raise ValueError(
            f"hidden_states and external_inputs must have the same number of "
            f"rows; got {T} vs {external_inputs.shape[0]}."
        )
    if T < 3:
        raise ValueError(
            "Trajectory must have at least 3 time steps to estimate CCI."
        )

    H_t = hidden_states[:-1]       # shape (T-1, d_h)
    H_t1 = hidden_states[1:]       # shape (T-1, d_h)
    E_t = external_inputs[:-1]     # shape (T-1, d_e)

    # Project hidden states to reduce dimensionality relative to sample count
    n_samples = H_t.shape[0]
    n_comp = min(pca_components, H_t.shape[1], n_samples - 2)
    if n_comp < H_t.shape[1]:
        H_t = _pca_project(H_t, n_comp)
        H_t1 = _pca_project(H_t1, n_comp)

    # Regularise external inputs against near-singular covariance
    n_comp_e = min(E_t.shape[1], n_samples - 2)
    if n_comp_e < E_t.shape[1]:
        E_t = _pca_project(E_t, n_comp_e)

    cci_val = gaussian_cmi(H_t, H_t1, E_t)
    return cci_val
