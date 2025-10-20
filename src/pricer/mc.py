# src/pricer/mc.py
import numpy as np
from scipy.stats import norm


def price_option(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float,
    m_paths: int = 100000,
):
    dW_T = np.sqrt(T) * np.random.standard_normal(m_paths)

    S_T = S0 * np.exp(T * (r - q - 0.5 * (sigma**2)) + sigma * dW_T)

    payoffs = np.maximum(S_T - K, 0)
    disc = np.exp(-r * T)
    discPay = disc * payoffs  # Discount back by the risk free rate

    SE = np.std(discPay, ddof=1) / np.sqrt(len(payoffs))
    CI = 1.96 * SE

    return [disc * payoffs.mean(), SE, CI]


def price_option_black(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float,
    t: float = 0.0,
):
    d1 = (np.log(S0 / K) + (r - q + (0.5 * (sigma**2))) * (T - t)) / (
        sigma * np.sqrt(T - t)
    )
    d2 = d1 - (sigma * np.sqrt((T - t)))

    C = S0 * np.exp((-q * (T - t))) * norm.cdf(d1) - K * np.exp(
        -r * (T - t)
    ) * norm.cdf(d2)
    return C
