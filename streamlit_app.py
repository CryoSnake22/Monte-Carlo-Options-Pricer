# app/streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.pricer.mc import price_euro_call_mc

st.set_page_config(page_title="Monte Carlo Option Pricer", layout="centered")
st.title("Monte Carlo European Call (GBM)")

col1, col2 = st.columns(2)
with col1:
    S0 = st.number_input("Spot (S)", 1.0, 1e6, 100.0, step=1.0)
    K = st.number_input("Strike (K)", 1.0, 1e6, 100.0, step=1.0)
    T = st.number_input("Maturity (years, T)", 0.01, 10.0, 1.0, step=0.01)
with col2:
    r = st.number_input(
        "Risk-free rate (r)", -0.05, 0.5, 0.02, step=0.005, format="%.4f"
    )
    sigma = st.number_input(
        "Volatility (σ)", 0.0001, 5.0, 0.2, step=0.01, format="%.4f"
    )

n_paths = st.slider("Paths", 1_000, 500_000, 50_000, step=1_000)
n_steps = st.slider("Steps per path", 50, 1000, 252, step=50)
antithetic = st.checkbox("Antithetic variates", True)

if st.button("Price option"):
    price, se = price_euro_call_mc(S0, K, T, r, sigma, n_paths, n_steps, antithetic)
    st.metric("Price (±1 SE)", f"{price:.4f}", f"± {se:.4f}")

    # Show payoff distribution sample
    # (Quick re-sim using same inputs just for the histogram)
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    S_T = S0 * np.exp(np.cumsum(drift + vol * Z, axis=1)[:, -1])
    payoff = np.maximum(S_T - K, 0.0) * np.exp(-r * T)

    fig, ax = plt.subplots()
    ax.hist(payoff, bins=60)
    ax.set_title("Discounted Payoff Distribution")
    ax.set_xlabel("Payoff")
    ax.set_ylabel("Frequency")
    st.pyplot(fig, clear_figure=True)
