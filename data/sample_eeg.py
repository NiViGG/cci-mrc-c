import numpy as np


def generate_synthetic_eeg(
    mode: str = "conscious-like",
    seq_len: int = 512,
    channels: int = 16,
    noise_scale: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """
    Reproducible synthetic EEG-like generator.
    Modes:
      - conscious-like
      - anesthesia-like
      - deep-noise
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, seq_len)

    if mode == "conscious-like":
        base = (
            0.8 * np.sin(8.0 * t)[:, None]
            + 0.5 * np.sin(13.0 * t + 0.3)[:, None]
            + 0.2 * np.sin(24.0 * t + 1.2)[:, None]
        )
    elif mode == "anesthesia-like":
        base = (
            1.0 * np.sin(2.0 * t)[:, None]
            + 0.2 * np.sin(4.0 * t + 0.5)[:, None]
            + 0.1 * np.sin(6.0 * t + 1.0)[:, None]
        )
    elif mode == "deep-noise":
        base = np.zeros((seq_len, 1), dtype=float)
    else:
        raise ValueError(f"Unknown EEG mode: {mode}")

    channel_mix = rng.normal(1.0, 0.1, size=(1, channels))
    noise = rng.normal(0.0, noise_scale, size=(seq_len, channels))
    signal = base @ np.ones((1, channels))
    return (signal * channel_mix + noise).astype(np.float32)
