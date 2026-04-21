import torch


class CCIEstimator:
    def __init__(self, jitter: float = 1e-6, max_jitter: float = 1e-4):
        self.jitter = jitter
        self.max_jitter = max_jitter

    @staticmethod
    def _as_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(1)
        return x

    @staticmethod
    def _cov(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        n = x.size(0)
        if n < 2:
            raise ValueError("Need at least two samples to estimate covariance.")
        return (x.T @ x) / (n - 1)

    def _logdet_cholesky_adaptive(
        self, cov: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        dim = cov.size(0)
        eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)

        jitter = self.jitter
        while jitter <= self.max_jitter:
            try:
                reg_cov = cov + eye * jitter
                chol = torch.linalg.cholesky(reg_cov)
                logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
                return logdet, jitter
            except RuntimeError:
                jitter *= 10.0

        raise RuntimeError(
            "Cholesky decomposition failed for all jitter values "
            f"in [{self.jitter:.1e}, {self.max_jitter:.1e}]."
        )

    def _entropy(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_2d(x)
        cov = self._cov(x)
        logdet, _ = self._logdet_cholesky_adaptive(cov)
        dim = cov.size(0)
        two_pi = torch.tensor(2.0 * torch.pi, device=cov.device, dtype=cov.dtype)
        return 0.5 * logdet + 0.5 * dim * (1.0 + torch.log(two_pi))

    def compute(self, h_t: torch.Tensor, h_t1: torch.Tensor, e_t: torch.Tensor) -> float:
        h_t = self._as_2d(h_t)
        h_t1 = self._as_2d(h_t1)
        e_t = self._as_2d(e_t)

        if not (h_t.size(0) == h_t1.size(0) == e_t.size(0)):
            raise ValueError("h_t, h_t1, and e_t must have matching sample counts.")

        xz = torch.cat([h_t, e_t], dim=1)
        yz = torch.cat([h_t1, e_t], dim=1)
        xyz = torch.cat([h_t, h_t1, e_t], dim=1)

        cmi = self._entropy(xz) + self._entropy(yz) - self._entropy(xyz) - self._entropy(e_t)
        return max(float(cmi.item()), 0.0)
