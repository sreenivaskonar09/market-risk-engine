# Multi-Asset Risk Engine
### FRTB-Compliant VaR / Expected Shortfall · $10M Mock Portfolio · 10-Year Simulation

---

## Overview

A production-grade quantitative risk system covering a three-asset-class portfolio:

| Asset | Instrument | Notional | Risk Factors |
|-------|-----------|----------|--------------|
| **Equity** | S&P 500 ATM Call Options | $5,000,000 | Delta, Gamma, Vega |
| **FX** | EUR/USD Forward | $3,000,000 | Delta (spot sensitivity) |
| **Bonds** | 10Y US Treasury (Sovereign) | $2,000,000 | Duration, DV01 |

---

## Project Structure

```
risk_engine/
├── data_generator.py   # Market data simulation (GBM + GARCH + Vasicek)
├── risk_engine.py      # VaR, ES, FRTB ES, Stress Tests, Backtest, PLA
├── db.py               # SQLite persistence layer
├── schema.sql          # Full relational schema (SQLite / PostgreSQL)
├── dashboard.py        # Interactive Dash dashboard
├── requirements.txt
└── README.md
```

---

## Methodology

### 1 · Data Generation
- **Equity**: Geometric Brownian Motion with GARCH(1,1) variance clustering
- **FX**: GBM with constant vol (EUR/USD major pair)
- **Bonds**: Vasicek mean-reverting yield model → price via `P = F·exp(-y·T)`
- **Greeks**: Black-Scholes ATM call option recomputed daily (delta, gamma, vega)

### 2 · Risk Engine

#### Historical Simulation VaR (99%)
```
VaR₀.₉₉ = quantile(PnL_historical, 1%)
```
Rolling 250 business-day window.

#### Expected Shortfall (97.5%)
```
ES₀.₉₇₅ = E[PnL | PnL ≤ VaR₀.₉₇₅]
```
Mean of all losses exceeding the 97.5th-percentile threshold.

#### FRTB Expected Shortfall (BCBS 352)
Liquidity Horizon (LH) scaling per asset class:

| Asset | Liquidity Horizon |
|-------|------------------|
| Equity Large Cap | 10 days |
| FX Major | 10 days |
| IG Sovereign Bonds | 20 days |

```
ES_FRTB = √( Σᵢ [ESᵢ · √(LHᵢ / LH_base)]² )
```
where `LH_base = 10`. This penalises less-liquid positions for holding-period risk.

### 3 · Stress Testing
Five macro scenarios with decomposed P&L attribution:

| Scenario | Equity Shock | Yield Shock | Trigger |
|----------|-------------|------------|---------|
| COVID-2020 Crash | -34% | -60bps | Pandemic lockdowns |
| GFC 2008 | -50% | -150bps | Lehman collapse |
| Banking Crisis 2023 | -12% | +80bps | SVB / Credit Suisse |
| Taper Tantrum 2013 | -6% | +130bps | Fed tapering signal |
| Rate Shock +300bps | -15% | +300bps | Regulatory parallel shift |

**Limit Breach**: flagged if `|Total P&L / Portfolio Value| > 15%`

### 4 · Backtesting

#### Kupiec POF (Proportion of Failures)
```
LR_POF = -2 ln[ p^n (1-p)^(T-n) / p̂^n (1-p̂)^(T-n) ] ~ χ²(1)
```
Basel Traffic Light zones:
- 🟢 **GREEN**: exception rate ≤ 1.0% (model acceptable)
- 🟡 **YELLOW**: 1.0% – 1.5% (under scrutiny)
- 🔴 **RED**: > 1.5% (model rejected, capital add-on required)

#### Christoffersen Independence Test
Tests whether exceptions cluster in time (a sign of model misspecification):
```
LR_IND = -2 ln[L(π) / L(π₀₁, π₁₁)] ~ χ²(1)
```

### 5 · P&L Attribution (PLA)
Compares the Greeks-based Risk-Theoretical P&L (RTPL) of the option book against its
full Black-Scholes revaluation Hypothetical P&L (HPL), using prior-day Greeks:
```
RTPL ≈ Δ·ΔS + ½·Γ·ΔS² + V·Δσ + Θ·Δt
HPL  = C(S₁, σ₁, T−Δt) − C(S₀, σ₀, T)
```
Pass criteria (FRTB IMA eligibility):
- Spearman correlation ≥ 0.80
- Variance ratio ∈ [0.80, 1.20]

---

## Setup & Running

```bash
# 1 · Install dependencies
pip install -r requirements.txt

# 2 · Generate data + run risk engine
python risk_engine.py

# 3 · Populate SQLite database
python db.py

# 4 · Launch dashboard
python dashboard.py
# Open http://localhost:8050
```

---

## Results (seed 42, 10-year simulation)

| Metric | Value |
|--------|-------|
| VaR 99% (latest 250 days) | −$329,245 |
| ES 97.5% | −$345,144 |
| FRTB ES (LH-adjusted) | −$353,358 |
| Diversification benefit | 13.6% |
| VaR exceptions | 38 / 2,271 days (1.67% vs 1.00% expected) |
| Kupiec POF | LR = 8.65, p = 0.003 → **RED zone** (model rejected) |
| Christoffersen independence | p = 0.67 → no significant clustering |
| PLA (RTPL vs HPL) | 100% of windows pass |

| Stress Scenario | Total P&L | % Capital | Limit Breach (>15%) |
|-----------------|----------:|----------:|:-------------------:|
| COVID-2020 Crash | −$1.44M | −14.4% | No |
| GFC 2008 | −$2.07M | −20.7% | **Yes** |
| Banking Crisis 2023 | −$0.57M | −5.7% | No |
| Taper Tantrum 2013 | −$0.56M | −5.6% | No |
| Rate Shock +300bps | −$1.19M | −11.9% | No |

**Interpretation:** the 250-day historical-simulation VaR underestimates tail risk.
The equity leg follows a GARCH(1,1) process, so a fixed equal-weighted window reacts
too slowly to volatility regime shifts. That produces excess exceptions and a Kupiec
rejection. The exceptions are not significantly clustered day-to-day, so the failure is
in coverage rather than independence. Natural next steps are filtered historical
simulation (FHS) or EWMA-weighted VaR.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Numerical | NumPy, SciPy (chi² tests, Black-Scholes) |
| Data | Pandas (rolling windows, time series) |
| Visualisation | Plotly, Dash, Dash Bootstrap |
| Database | SQLite (schema portable to PostgreSQL) |
| Risk Standards | FRTB (BCBS 352), Basel III/IV |
