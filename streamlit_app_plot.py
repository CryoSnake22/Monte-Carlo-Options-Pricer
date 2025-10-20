# app/streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.pricer.mc import price_option, price_option_black

plt.style.use("default")

st.set_page_config(page_title="Monte Carlo Option Pricer", layout="wide")
st.title("Monte Carlo European Call (GBM)")

# antithetic = st.checkbox("Antithetic variates", True)

# axes[0].set_title("random noise")
# axes[0].set_ylabel("value")
# axes[0].plot(Z.T)


# Latex
st.markdown(r"""
We use geometric brownian motion to simulate stock behaviour under a risk-neutral measure in order to get our expected payoff.
$$
S_{t}=S_{0}\exp\left( t\left( r-q-\frac{\sigma^{2}}{2} \right) +\sigma W_{t}\right) 
$$
This results in the following behaviour
""")


S0 = st.sidebar.number_input("Spot (S)", 1.0, 1e6, 100.0, step=1.0)
K = st.sidebar.number_input("Strike (K)", 1.0, 1e6, 100.0, step=1.0)
T = st.sidebar.number_input("Maturity (years, T)", 0.01, 10.0, 1.0, step=0.01)
r = st.sidebar.number_input(
    "Risk-free rate (r)", -0.05, 0.5, 0.02, step=0.005, format="%.4f"
)
sigma = st.sidebar.number_input(
    "Volatility (σ)", 0.0001, 5.0, 0.2, step=0.01, format="%.4f"
)
q = st.sidebar.number_input(
    "Yearly Dividends (q)", 0.00, 0.5, 0.00, step=0.005, format="%.4f"
)
col4, col5 = st.columns(2)
n_paths = st.sidebar.slider("Paths", 5, 50_000, 5000, step=1_000)
n_steps = st.sidebar.slider("Steps per path", 50, 1000, 252, step=50)

dt = T / n_steps

t = np.arange(1, n_steps + 1) * dt

fig, axes = plt.subplots(
    1,
    2,
    figsize=(20, 4),
)
# Demo for GBM

Z = np.random.standard_normal((n_paths, n_steps))

increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
S_t = S0 * np.cumprod(np.exp(increments), axis=1)

# dW = np.sqrt(dt) * Z
#
#
# W = np.cumsum(dW, axis=1)
# S_t = S0 * np.exp(t * (r - q - (0.5 * sigma**2)) + sigma * W)

axes[0].set_title("5 Simulated Stock Paths")
axes[0].set_ylabel("Value")
axes[0].plot(t, S_t[0:5].T)
axes[0].plot(t, [K] * len(t), "r--", label="Strike Price (K)")
axes[0].legend()

axes[1].set_title("40 Simulated Stock Paths")
axes[1].set_ylabel("value")
axes[1].plot(t, S_t[0:40].T, alpha=0.6)
axes[1].plot(t, [K] * len(t), "r--", label="Strike Price (K)")
axes[1].legend()

st.pyplot(fig, clear_figure=True)
price = price_option(S0, K, T, sigma, r, q, n_paths)
sterr = 0.00

# Pricing comparison
st.markdown(r"""
Now that we've simulated stock movement, we can attempt to price options. For a European style option, we only worry about the price at maturity $T$. In order to calculuate expected discounted payoff we have to folllow
$$
P=e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{payoff}(S_{T})]
$$
Where $\text{payoff}=\max (S_{T}-K,0)$. Using Monte Carlo methods to estimate $\mathbb{E}^{\mathbb{Q}}[\text{payoff}(S_{T})]$ yields our estimate value. We use the Black Scholes closed form to benchmark our model for accuracy. For more information on Black Scholes, refer to [my blog post on the subject](https://charlo.tech/posts/2025-10-15/). 

Following this methodology yields the following results:
""")


# Show payoff distribution sample
# (Quick re-sim using same inputs just for the histogram)

n = np.logspace(
    2, np.log10(n_paths), num=50, dtype=int
)  # logspace for proper convergence shown
estimates = [price_option(S0, K, T, sigma, r, q, m_paths) for m_paths in n]

fig2, ax = plt.subplots(figsize=(20, 4))
bs_estimate = price_option_black(S0, K, T, sigma, r, q)
bs_line = np.full_like(estimates, bs_estimate)


price1, price2 = st.columns(2)
with price1:
    st.metric("Our estimate", f"{price:.4f}", f"± {sterr}")
with price2:
    st.metric("Black Scholes estimate", f"{bs_estimate:.4f}")


ax.set_ylabel("Option Price")
ax.set_xlabel("Number of Iterations")
ax.set_title("Convergence of monte carlo estimate")
ax.plot(n, bs_line, "r--", label="Black Scholes Pricing")
ax.plot(n, estimates, label="monte carlo estimate")
# plt.grid(True, which="both", ls="--", lw=0.5)
ax.legend()
st.pyplot(fig2, clear_figure=True)

st.markdown(
    r"""
    Increasing the number of paths essentially increases our number of samples which makes the estimate closer to the true value. 
    Try playing with the parameters and see what happens.
    """
)
