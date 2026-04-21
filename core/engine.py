import torch


class CCIEngine:
    def __init__(self, jitter: float = 1e-7, max_jitter: float = 1e-4):
        self.jitter = jitter
        self.max_jitter = max_jitter
        self.last_jitter: float | None = None

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
        cov = (x.T @ x) / (n - 1)
        return 0.5 * (cov + cov.T)

    def _logdet_adaptive(self, cov: torch.Tensor) -> tuple[torch.Tensor, float]:
        dim = cov.size(0)
        eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
        jitter = self.jitter
        while jitter <= self.max_jitter:
            try:
                chol = torch.linalg.cholesky(cov + eye * jitter)
                logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
                self.last_jitter = jitter
                return logdet, jitter
            except RuntimeError:
                jitter *= 10.0
        raise RuntimeError("Cholesky failed for all jitter values in configured range.")

    def _entropy(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_2d(x)
        cov = self._cov(x)
        logdet, _ = self._logdet_adaptive(cov)
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

    def process_signal(self, hidden_states: torch.Tensor, eeg_signal: torch.Tensor) -> dict:
        hidden_states = self._as_2d(hidden_states)
        eeg_signal = self._as_2d(eeg_signal)
        if hidden_states.size(0) < 3:
            raise ValueError("Need at least 3 hidden-state samples for CCI process.")
        if eeg_signal.size(0) < hidden_states.size(0):
            raise ValueError("eeg_signal must be at least as long as hidden_states.")

        h_t = hidden_states[:-1]
        h_t1 = hidden_states[1:]
        e_t = eeg_signal[: h_t.size(0)]
        value = self.compute(h_t, h_t1, e_t)
        return {
            "cci_bits": value,
            "used_jitter": self.last_jitter,
            "samples": int(h_t.size(0)),
        }
