# src/pricer/mc.py
import numpy as np


def price_euro_call_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 50_000,
    n_steps: int = 252,
    antithetic: bool = True,
):
    """Risk-neutral MC for a European call; returns (price, stderr)."""
    dt = T / n_steps
    paths = n_paths if not antithetic else n_paths // 2

    Z = np.random.standard_normal((paths, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])  # doubles path count

    # GBM exact discretization
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    increments = drift + vol * Z
    log_S = np.log(S0) + np.cumsum(increments, axis=1)
    S_T = np.exp(log_S[:, -1])

    payoff = np.maximum(S_T - K, 0.0)
    disc = np.exp(-r * T)
    price_paths = disc * payoff

    price = price_paths.mean()
    stderr = price_paths.std(ddof=1) / np.sqrt(price_paths.size)
    return price, stderr
