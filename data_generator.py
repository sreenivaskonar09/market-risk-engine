"""
data_generator.py
Simulates 10 years of daily market data for a multi-asset portfolio:
  - Equity Options (S&P 500 proxy)
  - FX Forwards (EUR/USD)
  - Sovereign Bonds (10Y US Treasury)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Parameters ────────────────────────────────────────────────────────────────
TRADING_DAYS = 2520          # ~10 years
START_DATE   = "2014-01-02"

# Equity (SPY proxy)
EQ_S0    = 1850.0            # starting index level
EQ_MU    = 0.08 / 252        # daily drift
EQ_SIGMA = 0.15 / np.sqrt(252)

# FX (EUR/USD)
FX_S0    = 1.3600
FX_MU    = 0.00 / 252
FX_SIGMA = 0.07 / np.sqrt(252)

# Bond (10Y yield → price)
BOND_Y0     = 0.025          # starting yield
BOND_KAPPA  = 0.10           # mean reversion speed
BOND_THETA  = 0.025          # long-run mean
BOND_SIGMA_Y= 0.008 / np.sqrt(252)  # yield vol
BOND_DURATION = 8.5          # modified duration (years)
BOND_FACE    = 1000.0

# Portfolio notionals (summing to ~$10M)
EQ_NOTIONAL   = 5_000_000   # $5M in equity options
FX_NOTIONAL   = 3_000_000   # $3M in FX forwards
BOND_NOTIONAL = 2_000_000   # $2M in sovereign bonds


def simulate_equity(n: int) -> np.ndarray:
    """Geometric Brownian Motion with volatility clustering (GARCH-like)."""
    shocks = np.random.normal(0, 1, n)
    # simple GARCH(1,1) variance path
    var = np.zeros(n)
    var[0] = EQ_SIGMA ** 2
    alpha, beta = 0.08, 0.88
    omega = EQ_SIGMA**2 * (1 - alpha - beta)
    for t in range(1, n):
        var[t] = omega + alpha * (shocks[t-1] * np.sqrt(var[t-1]))**2 + beta * var[t-1]
    returns = EQ_MU + np.sqrt(var) * shocks
    prices = EQ_S0 * np.exp(np.cumsum(returns))
    return prices, returns


def simulate_fx(n: int) -> np.ndarray:
    shocks = np.random.normal(0, 1, n)
    returns = FX_MU + FX_SIGMA * shocks
    prices = FX_S0 * np.exp(np.cumsum(returns))
    return prices, returns


def simulate_bond_yields(n: int) -> np.ndarray:
    """Vasicek mean-reverting yield process."""
    y = np.zeros(n)
    y[0] = BOND_Y0
    for t in range(1, n):
        dy = BOND_KAPPA * (BOND_THETA - y[t-1]) + BOND_SIGMA_Y * np.random.normal()
        y[t] = max(y[t-1] + dy, 0.001)  # floor at 10bps
    # Bond price ≈ Face * exp(-y * T) for zero-coupon approximation (T=10)
    T = 10
    prices = BOND_FACE * np.exp(-y * T)
    returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    return prices, returns, y


def compute_greeks(eq_prices: np.ndarray, iv_base: float = 0.20) -> pd.DataFrame:
    """
    Compute Black-Scholes Greeks for an ATM call option on each day.
    Strike is reset monthly to stay near ATM.
    """
    T = 30 / 252          # 30-day option
    r = 0.04              # risk-free rate
    rows = []
    for i, S in enumerate(eq_prices):
        K = S  # ATM
        iv = iv_base + 0.05 * np.sin(i / 63)  # slight IV cycle
        d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
        vega  = S * norm.pdf(d1) * np.sqrt(T)
        rows.append({"delta": delta, "gamma": gamma, "vega": vega, "iv": iv})
    return pd.DataFrame(rows)


def build_portfolio() -> pd.DataFrame:
    dates = pd.bdate_range(START_DATE, periods=TRADING_DAYS)

    eq_px,   eq_ret   = simulate_equity(TRADING_DAYS)
    fx_px,   fx_ret   = simulate_fx(TRADING_DAYS)
    bond_px, bond_ret, yields = simulate_bond_yields(TRADING_DAYS)

    greeks = compute_greeks(eq_px)

    # Daily P&L per asset (dollar terms)
    eq_units   = EQ_NOTIONAL   / eq_px[0]
    fx_units   = FX_NOTIONAL   / fx_px[0]
    bond_units = BOND_NOTIONAL / bond_px[0]

    eq_pnl   = eq_units   * (eq_px   * eq_ret)
    fx_pnl   = fx_units   * (fx_px   * fx_ret)
    bond_pnl = bond_units * (bond_px * bond_ret)
    total_pnl = eq_pnl + fx_pnl + bond_pnl

    df = pd.DataFrame({
        "date":        dates,
        # Prices
        "eq_price":   eq_px,
        "fx_price":   fx_px,
        "bond_price": bond_px,
        "bond_yield": yields,
        # Returns
        "eq_ret":     eq_ret,
        "fx_ret":     fx_ret,
        "bond_ret":   bond_ret,
        # Greeks
        "delta":      greeks["delta"].values,
        "gamma":      greeks["gamma"].values,
        "vega":       greeks["vega"].values,
        "impl_vol":   greeks["iv"].values,
        # P&L
        "eq_pnl":     eq_pnl,
        "fx_pnl":     fx_pnl,
        "bond_pnl":   bond_pnl,
        "total_pnl":  total_pnl,
    })
    df.set_index("date", inplace=True)
    return df


if __name__ == "__main__":
    df = build_portfolio()
    df.to_csv("portfolio_data.csv")
    print(f"Generated {len(df)} rows | P&L range: "
          f"${df['total_pnl'].min():,.0f} / ${df['total_pnl'].max():,.0f}")
    print(df.tail(3))
