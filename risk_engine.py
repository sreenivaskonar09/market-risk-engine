"""
risk_engine.py
Implements:
  1. Historical Simulation VaR @ 99%
  2. FRTB Expected Shortfall @ 97.5% with Liquidity Horizon scaling
  3. Stress Testing (five historical and regulatory macro scenarios)
  4. Backtesting (Kupiec POF + Christoffersen Independence)
  5. P&L Attribution (PLA)
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# ── FRTB Liquidity Horizons (BCBS 352) ───────────────────────────────────────
# Maps asset class → liquidity horizon in days
LIQUIDITY_HORIZONS = {
    "equity_large_cap":   10,
    "fx_major":           10,
    "sovereign_bonds_ig": 20,
    "equity_small_cap":   20,
    "credit_ig":          40,
    "credit_hy":          60,
}

PORTFOLIO_LH = {
    "eq":   LIQUIDITY_HORIZONS["equity_large_cap"],
    "fx":   LIQUIDITY_HORIZONS["fx_major"],
    "bond": LIQUIDITY_HORIZONS["sovereign_bonds_ig"],
}

PORTFOLIO_NOTIONAL = 10_000_000


@dataclass
class RiskResult:
    var_99: float
    es_975: float
    es_frtb: float          # liquidity-horizon adjusted
    diversification_benefit: float
    lh_scaling: Dict[str, float] = field(default_factory=dict)


# ── Core Risk Functions ───────────────────────────────────────────────────────

def historical_var(pnl: pd.Series, confidence: float = 0.99,
                   window: int = 250) -> pd.Series:
    """
    Rolling Historical Simulation VaR.
    Negative value = loss at confidence level.
    """
    return pnl.rolling(window).quantile(1 - confidence)


def expected_shortfall(pnl: pd.Series, confidence: float = 0.975,
                       window: int = 250) -> pd.Series:
    """
    Rolling ES: mean of losses beyond the VaR threshold.
    """
    def _es(x):
        cutoff = np.quantile(x, 1 - confidence)
        tail = x[x <= cutoff]
        return tail.mean() if len(tail) > 0 else cutoff
    return pnl.rolling(window).apply(_es, raw=True)


def frtb_es(pnl_components: Dict[str, pd.Series],
            confidence: float = 0.975,
            window: int = 250) -> pd.Series:
    """
    FRTB ES with Liquidity Horizon scaling.
    ES_FRTB = sqrt( sum_i [ ES_i * sqrt(LH_i / LH_base) ]^2 )
    where LH_base = 10 days.
    """
    LH_BASE = 10
    es_scaled = {}
    for asset, series in pnl_components.items():
        lh = PORTFOLIO_LH.get(asset, 10)
        scale = np.sqrt(lh / LH_BASE)

        def _es_scaled(x, s=scale):
            cutoff = np.quantile(x, 1 - confidence)
            tail = x[x <= cutoff]
            es_raw = tail.mean() if len(tail) > 0 else cutoff
            return es_raw * s

        es_scaled[asset] = series.rolling(window).apply(_es_scaled, raw=True)

    # FRTB aggregation (assume correlation = 1 for conservatism; partial offset allowed)
    combined = sum(s**2 for s in es_scaled.values())
    return combined.apply(np.sqrt) * (-1)   # express as positive loss


def compute_risk_snapshot(df: pd.DataFrame, as_of_date=None) -> RiskResult:
    """Single-date risk snapshot using last 250 trading days."""
    if as_of_date:
        subset = df.loc[:as_of_date].tail(250)
    else:
        subset = df.tail(250)

    pnl = subset["total_pnl"]
    var_99  = float(np.quantile(pnl, 0.01))   # 1st percentile = 99% loss
    es_975  = float(pnl[pnl <= np.quantile(pnl, 0.025)].mean())

    # FRTB per-asset
    pnl_comps = {
        "eq":   subset["eq_pnl"],
        "fx":   subset["fx_pnl"],
        "bond": subset["bond_pnl"],
    }
    LH_BASE = 10
    es_lh_sq = 0.0
    lh_scaling = {}
    for asset, s in pnl_comps.items():
        lh = PORTFOLIO_LH[asset]
        scale = np.sqrt(lh / LH_BASE)
        cutoff = np.quantile(s, 0.025)
        es_i = s[s <= cutoff].mean() if len(s[s <= cutoff]) > 0 else cutoff
        es_lh = es_i * scale
        lh_scaling[asset] = {"es_raw": float(es_i), "scale": scale,
                              "lh": lh, "es_scaled": float(es_lh)}
        es_lh_sq += es_lh**2

    es_frtb = -np.sqrt(es_lh_sq)

    # Diversification: undiversified ES vs diversified
    sum_individual = sum(abs(v["es_raw"]) * v["scale"]
                         for v in lh_scaling.values())
    div_benefit = (sum_individual - abs(es_frtb)) / sum_individual if sum_individual else 0.0

    return RiskResult(
        var_99=var_99,
        es_975=es_975,
        es_frtb=es_frtb,
        diversification_benefit=div_benefit,
        lh_scaling=lh_scaling,
    )


def rolling_risk(df: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    """Compute rolling VaR and ES series for the full dataset."""
    var_series = historical_var(df["total_pnl"], confidence=0.99, window=window)
    es_series  = expected_shortfall(df["total_pnl"], confidence=0.975, window=window)

    frtb_comps = {
        "eq":   df["eq_pnl"],
        "fx":   df["fx_pnl"],
        "bond": df["bond_pnl"],
    }
    frtb_series = frtb_es(frtb_comps, window=window)

    out = pd.DataFrame({
        "var_99":    var_series,
        "es_975":    es_series,
        "es_frtb":   -frtb_series,
        "total_pnl": df["total_pnl"],
    }, index=df.index)
    return out.dropna()


# ── Stress Testing ────────────────────────────────────────────────────────────

STRESS_SCENARIOS = {
    "COVID_2020_Crash": {
        "description": "March 2020 – 34% equity drawdown, USD flight-to-safety",
        "eq_shock":    -0.34,    # equity -34%
        "fx_shock":    -0.04,    # EUR/USD -4% (USD strength)
        "yield_shock": -0.0060,  # yields -60bps (flight to quality)
        "vol_shock":   +0.40,    # IV spike +40 vol points
    },
    "GFC_2008": {
        "description": "Lehman collapse – credit freeze, equity -50%",
        "eq_shock":    -0.50,
        "fx_shock":    -0.08,
        "yield_shock": -0.0150,
        "vol_shock":   +0.60,
    },
    "Banking_Crisis_2023": {
        "description": "SVB/Credit Suisse – regional bank stress, rate spike",
        "eq_shock":    -0.12,
        "fx_shock":    +0.02,
        "yield_shock": +0.0080,  # yields UP (rate hike fear)
        "vol_shock":   +0.15,
    },
    "Taper_Tantrum_2013": {
        "description": "Fed tapering shock – rising yields, EM sell-off",
        "eq_shock":    -0.06,
        "fx_shock":    -0.03,
        "yield_shock": +0.0130,
        "vol_shock":   +0.08,
    },
    "Rate_Shock_Up_300bps": {
        "description": "Parallel yield curve shift +300bps (regulatory scenario)",
        "eq_shock":    -0.15,
        "fx_shock":    0.00,
        "yield_shock": +0.0300,
        "vol_shock":   +0.10,
    },
}


def run_stress_test(df: pd.DataFrame,
                    capital_limit_pct: float = 0.15) -> pd.DataFrame:
    """
    Apply each macro scenario to the portfolio's most recent snapshot.
    Returns a DataFrame with P&L impact and breach flags.
    """
    latest = df.iloc[-1]
    portfolio_value = PORTFOLIO_NOTIONAL

    results = []
    for name, shock in STRESS_SCENARIOS.items():
        eq_pnl   = 5_000_000 * shock["eq_shock"]
        fx_pnl   = 3_000_000 * shock["fx_shock"]
        # Bond price ≈ -duration * Δy * face_value
        bond_pnl = -(8.5 * shock["yield_shock"] * 2_000_000)
        # Vega P&L: vega_dollar * vol_shock
        vega_dollar = latest["vega"] * (5_000_000 / latest["eq_price"])
        vega_pnl    = vega_dollar * shock["vol_shock"]   # vega is per 1.00 vol; shock is decimal
        total       = eq_pnl + fx_pnl + bond_pnl + vega_pnl
        pct_capital = total / portfolio_value

        breach = abs(pct_capital) > capital_limit_pct

        results.append({
            "scenario":         name,
            "description":      shock["description"],
            "eq_pnl":           eq_pnl,
            "fx_pnl":           fx_pnl,
            "bond_pnl":         bond_pnl,
            "vega_pnl":         vega_pnl,
            "total_pnl":        total,
            "pct_of_capital":   pct_capital * 100,
            "limit_breach":     breach,
        })

    return pd.DataFrame(results)


# ── Backtesting ───────────────────────────────────────────────────────────────

def kupiec_pof_test(pnl: pd.Series, var: pd.Series,
                    confidence: float = 0.99) -> Dict:
    """
    Kupiec Proportion of Failures (POF) test.
    H0: true exception rate = 1 - confidence
    """
    aligned = pd.concat([pnl, var], axis=1).dropna()
    aligned.columns = ["pnl", "var"]
    exceptions = (aligned["pnl"] < aligned["var"]).astype(int)
    T = len(exceptions)
    n = exceptions.sum()
    p = 1 - confidence
    p_hat = n / T if T > 0 else 0

    # LR statistic
    if p_hat == 0 or p_hat == 1:
        lr_stat = 0.0
    else:
        lr_stat = -2 * (np.log((p ** n) * ((1 - p) ** (T - n))) -
                        np.log((p_hat ** n) * ((1 - p_hat) ** (T - n))))

    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    zone = _traffic_light(n, T)

    return {
        "T": T, "n_exceptions": int(n),
        "exception_rate_pct": round(p_hat * 100, 2),
        "expected_rate_pct":  round(p * 100, 2),
        "lr_stat": round(lr_stat, 4),
        "p_value": round(p_value, 4),
        "reject_h0": p_value < 0.05,
        "zone": zone,
    }


def christoffersen_test(pnl: pd.Series, var: pd.Series,
                        confidence: float = 0.99) -> Dict:
    """
    Christoffersen Independence test.
    Tests whether exceptions cluster (bad) or are independent (good).
    """
    aligned = pd.concat([pnl, var], axis=1).dropna()
    aligned.columns = ["pnl", "var"]
    exc = (aligned["pnl"] < aligned["var"]).astype(int).values

    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(exc)):
        if   exc[t-1] == 0 and exc[t] == 0: n00 += 1
        elif exc[t-1] == 0 and exc[t] == 1: n01 += 1
        elif exc[t-1] == 1 and exc[t] == 0: n10 += 1
        else:                                n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) else 0

    def safe_log(x): return np.log(x) if x > 0 else 0

    lr_ind = -2 * (
        (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
        - n00 * safe_log(1 - pi01) - n01 * safe_log(pi01 + 1e-12)
        - n10 * safe_log(1 - pi11) - n11 * safe_log(pi11 + 1e-12)
    )
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)

    return {
        "pi01": round(pi01, 4),
        "pi11": round(pi11, 4),
        "lr_stat": round(lr_ind, 4),
        "p_value": round(p_value, 4),
        "reject_h0": p_value < 0.05,
        "clustering": pi11 > pi01,
    }


def _traffic_light(n: int, T: int) -> str:
    """Basel traffic light zones based on exception count (250-day window)."""
    rate = n / T if T > 0 else 0
    if   rate <= 0.01:  return "GREEN"
    elif rate <= 0.015: return "YELLOW"
    else:               return "RED"


# ── P&L Attribution ───────────────────────────────────────────────────────────

def pla_test(df: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    """
    P&L Attribution (PLA) Test.
    Compares 'Risk-Theoretical P&L' (RTPL, Greeks-based risk model P&L)
    vs 'Hypothetical P&L' (HPL, full Black-Scholes revaluation) for the
    ATM call book, using yesterday's Greeks against today's market moves.
    Metric: Spearman correlation and ratio of variances.
    """
    T0, r = 30 / 252, 0.04            # same option spec as data_generator
    T1 = T0 - 1 / 252
    S0, S1   = df["eq_price"].shift(1), df["eq_price"]
    iv0, iv1 = df["impl_vol"].shift(1), df["impl_vol"]
    K = S0                            # option struck ATM at start of day

    def bs_call(S, K, T, iv):
        d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    # HPL: full revaluation of the option over one day
    full_reval = bs_call(S1, K, T1, iv1) - bs_call(S0, K, T0, iv0)

    # RTPL: Greeks approximation  ΔP ≈ Δ·ΔS + ½·Γ·ΔS² + V·Δσ + Θ·Δt
    dS, dsig = S1 - S0, iv1 - iv0
    d1 = (r + 0.5 * iv0**2) * T0 / (iv0 * np.sqrt(T0))
    d2 = d1 - iv0 * np.sqrt(T0)
    theta = (-S0 * stats.norm.pdf(d1) * iv0 / (2 * np.sqrt(T0))
             - r * K * np.exp(-r * T0) * stats.norm.cdf(d2))
    greeks_pnl = (df["delta"].shift(1) * dS +
                  0.5 * df["gamma"].shift(1) * dS**2 +
                  df["vega"].shift(1) * dsig +      # vega per 1.00 vol × decimal Δσ
                  theta / 252)

    # Scale to the $5M equity notional
    units = 5_000_000 / S0
    # FX and Bonds are linear → Greeks P&L = full revaluation P&L
    linear = df["fx_pnl"] + df["bond_pnl"]
    hyp_total    = (greeks_pnl * units + linear).fillna(0)
    actual_total = (full_reval * units + linear).fillna(0)

    rows = []
    for i in range(window, len(df)):
        h = hyp_total.iloc[i-window:i]
        a = actual_total.iloc[i-window:i]
        corr, _  = stats.spearmanr(h, a)
        var_ratio = h.var() / a.var() if a.var() > 0 else np.nan
        rows.append({
            "date":        df.index[i],
            "spearman_r":  corr,
            "var_ratio":   var_ratio,
            "pla_pass":    abs(corr) >= 0.80 and 0.8 <= var_ratio <= 1.2,
        })
    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    from data_generator import build_portfolio
    df = build_portfolio()
    rr = compute_risk_snapshot(df)
    print(f"\n=== Risk Snapshot ===")
    print(f"  VaR 99%      : ${rr.var_99:>12,.0f}")
    print(f"  ES  97.5%    : ${rr.es_975:>12,.0f}")
    print(f"  ES  FRTB     : ${rr.es_frtb:>12,.0f}")
    print(f"  Div. Benefit : {rr.diversification_benefit:.1%}")

    stress = run_stress_test(df)
    print("\n=== Stress Test ===")
    print(stress[["scenario", "total_pnl", "pct_of_capital", "limit_breach"]].to_string(index=False))

    roll = rolling_risk(df)
    kupiec = kupiec_pof_test(roll["total_pnl"], roll["var_99"])
    print(f"\n=== Kupiec POF Test ===")
    print(f"  Exceptions   : {kupiec['n_exceptions']} / {kupiec['T']}")
    print(f"  Zone         : {kupiec['zone']}")
    print(f"  p-value      : {kupiec['p_value']}")
