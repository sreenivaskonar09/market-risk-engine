"""
dashboard.py – Multi-Asset Risk Engine Dashboard
Run:  python dashboard.py
Open: http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ── Bootstrap Data ────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_generator import build_portfolio
from risk_engine import (
    rolling_risk, compute_risk_snapshot, run_stress_test,
    kupiec_pof_test, christoffersen_test, pla_test
)

print("Generating portfolio data...")
df      = build_portfolio()
roll    = rolling_risk(df)
stress  = run_stress_test(df)
pla     = pla_test(df)
snap    = compute_risk_snapshot(df)
kupiec  = kupiec_pof_test(roll["total_pnl"], roll["var_99"])
chris   = christoffersen_test(roll["total_pnl"], roll["var_99"])
print("Done.")

PORTFOLIO_VALUE = 10_000_000

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#0a0e1a",
    "panel":    "#111827",
    "border":   "#1e2d45",
    "accent1":  "#00d4ff",
    "accent2":  "#ff6b6b",
    "accent3":  "#ffd166",
    "green":    "#06d6a0",
    "text":     "#e2e8f0",
    "muted":    "#64748b",
    "eq":       "#00d4ff",
    "fx":       "#a78bfa",
    "bond":     "#ffd166",
}

FONT = "IBM Plex Mono"

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family=FONT, color=C["text"], size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
)

# ── Helper: kpi_card ──────────────────────────────────────────────────────────
def kpi(label, value, color=C["accent1"], sub=None):
    return dbc.Col(
        html.Div([
            html.P(label, style={"color": C["muted"], "fontSize": "0.70rem",
                                 "marginBottom": "4px", "letterSpacing": "0.08em",
                                 "textTransform": "uppercase"}),
            html.H3(value, style={"color": color, "fontFamily": FONT,
                                  "fontSize": "1.5rem", "margin": 0}),
            html.P(sub or "", style={"color": C["muted"], "fontSize": "0.68rem",
                                     "marginTop": "2px"})
        ], style={
            "background": C["panel"], "border": f"1px solid {C['border']}",
            "borderTop": f"2px solid {color}",
            "padding": "16px 20px", "borderRadius": "6px",
        }),
        md=3, sm=6, xs=12
    )


# ── Figure: Rolling VaR / ES ──────────────────────────────────────────────────
def fig_var_es():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll.index, y=roll["total_pnl"],
        name="Daily P&L", line=dict(color=C["muted"], width=0.8),
        fill="tozeroy", fillcolor="rgba(100,116,139,0.08)"))
    fig.add_trace(go.Scatter(x=roll.index, y=roll["var_99"],
        name="VaR 99%", line=dict(color=C["accent3"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=roll.index, y=roll["es_975"],
        name="ES 97.5%", line=dict(color=C["accent2"], width=1.5)))
    fig.add_trace(go.Scatter(x=roll.index, y=roll["es_frtb"],
        name="ES FRTB (LH-adj)", line=dict(color=C["accent1"], width=2)))

    # Mark exceptions
    exceptions = roll[roll["total_pnl"] < roll["var_99"]]
    fig.add_trace(go.Scatter(x=exceptions.index, y=exceptions["total_pnl"],
        mode="markers", name="VaR Exception",
        marker=dict(color=C["accent2"], size=5, symbol="x")))

    fig.update_layout(**LAYOUT_BASE, title="Rolling VaR (99%) · ES (97.5%) · FRTB ES",
                      yaxis=dict(tickformat="$,.0f", gridcolor=C["border"],
                                 zeroline=True, zerolinecolor=C["border"]),
                      xaxis=dict(gridcolor=C["border"]))
    return fig


# ── Figure: P&L Histogram with VaR line ──────────────────────────────────────
def fig_pnl_hist():
    pnl = df["total_pnl"].values
    var_val = np.quantile(pnl, 0.01)
    es_val  = pnl[pnl <= np.quantile(pnl, 0.025)].mean()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl, nbinsx=80,
        marker=dict(color=C["accent1"], opacity=0.6,
                    line=dict(color=C["border"], width=0.3)),
        name="P&L Distribution"))
    fig.add_vline(x=var_val, line=dict(color=C["accent3"], width=2, dash="dash"),
                  annotation_text=f"VaR 99%<br>${var_val:,.0f}",
                  annotation_font=dict(color=C["accent3"], size=10))
    fig.add_vline(x=es_val, line=dict(color=C["accent2"], width=2),
                  annotation_text=f"ES 97.5%<br>${es_val:,.0f}",
                  annotation_font=dict(color=C["accent2"], size=10),
                  annotation_position="top left")

    # Shade tail
    tail_x = pnl[pnl <= var_val]
    fig.add_trace(go.Histogram(x=tail_x, nbinsx=20,
        marker=dict(color=C["accent2"], opacity=0.9),
        name="Tail Losses (>VaR)"))

    fig.update_layout(**LAYOUT_BASE, title="P&L Distribution · 10-Year Full History",
                      xaxis=dict(tickformat="$,.0f", gridcolor=C["border"]),
                      yaxis=dict(gridcolor=C["border"], title="Frequency"),
                      barmode="overlay")
    return fig


# ── Figure: Asset Contribution ────────────────────────────────────────────────
def fig_asset_contrib():
    monthly = df[["eq_pnl", "fx_pnl", "bond_pnl"]].resample("ME").sum()
    fig = go.Figure()
    for col, name, color in [("eq_pnl","Equity Options",C["eq"]),
                               ("fx_pnl","FX Forwards",C["fx"]),
                               ("bond_pnl","Sovereign Bonds",C["bond"])]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col],
                             name=name, marker_color=color, opacity=0.85))
    fig.update_layout(**LAYOUT_BASE, title="Monthly P&L Attribution by Asset Class",
                      barmode="relative",
                      yaxis=dict(tickformat="$,.0f", gridcolor=C["border"]),
                      xaxis=dict(gridcolor=C["border"]))
    return fig


# ── Figure: Greeks Over Time ──────────────────────────────────────────────────
def fig_greeks():
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Delta", "Gamma", "Vega (IV)"),
                        vertical_spacing=0.06)
    for i, (col, color) in enumerate([("delta", C["eq"]),
                                        ("gamma", C["accent3"]),
                                        ("impl_vol", C["accent2"])], 1):
        fig.add_trace(go.Scatter(x=df.index, y=df[col],
                                  line=dict(color=color, width=1),
                                  name=col.title()), row=i, col=1)
    fig.update_layout(**LAYOUT_BASE, title="Option Greeks Trajectory",
                      height=360, showlegend=False)
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=C["border"], row=i, col=1)
        fig.update_yaxes(gridcolor=C["border"], row=i, col=1)
    return fig


# ── Figure: Stress Test ───────────────────────────────────────────────────────
def fig_stress():
    s = stress.copy()
    s["color"] = s["limit_breach"].map({True: C["accent2"], False: C["green"]})

    fig = go.Figure()
    for _, row in s.iterrows():
        for asset, col, c in [("Equity", "eq_pnl", C["eq"]),
                               ("FX", "fx_pnl", C["fx"]),
                               ("Bond", "bond_pnl", C["bond"]),
                               ("Vega", "vega_pnl", C["accent3"])]:
            fig.add_trace(go.Bar(
                x=[row["scenario"]], y=[row[col]],
                name=asset, marker_color=c,
                showlegend=(row.name == 0),
            ))
    fig.update_layout(**LAYOUT_BASE, title="Stress Test P&L Decomposition",
                      barmode="relative",
                      yaxis=dict(tickformat="$,.0f", gridcolor=C["border"]),
                      xaxis=dict(gridcolor=C["border"]))
    return fig


# ── Figure: Backtesting ───────────────────────────────────────────────────────
def fig_backtest():
    exc = roll["total_pnl"] < roll["var_99"]
    exc_dates = roll.index[exc]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll.index, y=roll["total_pnl"],
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
        line=dict(color=C["muted"], width=0.8), name="P&L"))
    fig.add_trace(go.Scatter(x=roll.index, y=roll["var_99"],
        line=dict(color=C["accent3"], width=1.5, dash="dot"), name="VaR 99%"))
    fig.add_trace(go.Scatter(
        x=exc_dates, y=roll.loc[exc_dates, "total_pnl"],
        mode="markers", name="Exception",
        marker=dict(color=C["accent2"], size=7, symbol="circle-open",
                    line=dict(width=2, color=C["accent2"]))))

    fig.update_layout(**LAYOUT_BASE, title="Backtesting: VaR Exceptions (250-day rolling)",
                      yaxis=dict(tickformat="$,.0f", gridcolor=C["border"]),
                      xaxis=dict(gridcolor=C["border"]))
    return fig


# ── Figure: PLA ──────────────────────────────────────────────────────────────
def fig_pla():
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Spearman Correlation (PLA)", "Variance Ratio"))
    fig.add_trace(go.Scatter(x=pla.index, y=pla["spearman_r"],
        line=dict(color=C["accent1"], width=1.5), name="Spearman r"), row=1, col=1)
    fig.add_hline(y=0.80, line=dict(color=C["green"], dash="dash"), row=1, col=1,
                  annotation_text="Pass Threshold")
    fig.add_trace(go.Scatter(x=pla.index, y=pla["var_ratio"],
        line=dict(color=C["accent3"], width=1.5), name="Var Ratio"), row=1, col=2)
    fig.add_hrect(y0=0.8, y1=1.2, row=1, col=2,
                  fillcolor=C["green"], opacity=0.1, line_width=0,
                  annotation_text="Pass Band")
    fig.update_layout(**LAYOUT_BASE, title="P&L Attribution (PLA) Validation",
                      showlegend=False, height=300)
    for c in [1, 2]:
        fig.update_xaxes(gridcolor=C["border"], row=1, col=c)
        fig.update_yaxes(gridcolor=C["border"], row=1, col=c)
    return fig


# ── Stress Table ──────────────────────────────────────────────────────────────
def stress_table_data():
    rows = []
    for _, r in stress.iterrows():
        rows.append({
            "Scenario":       r["scenario"],
            "Description":    r["description"],
            "Equity P&L":     f"${r['eq_pnl']:>12,.0f}",
            "FX P&L":         f"${r['fx_pnl']:>10,.0f}",
            "Bond P&L":       f"${r['bond_pnl']:>10,.0f}",
            "Total P&L":      f"${r['total_pnl']:>12,.0f}",
            "% Capital":      f"{r['pct_of_capital']:+.1f}%",
            "Breach":         "⚠ BREACH" if r["limit_breach"] else "✓ OK",
        })
    return rows


# ── App Layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        f"https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap",
    ],
    title="Multi-Asset Risk Engine"
)

zone_color = {"GREEN": C["green"], "YELLOW": C["accent3"], "RED": C["accent2"]}

app.layout = dbc.Container([

    # ── Header ──
    html.Div([
        html.H1("MULTI-ASSET RISK ENGINE", style={
            "fontFamily": FONT, "fontSize": "1.2rem", "letterSpacing": "0.2em",
            "color": C["accent1"], "margin": 0
        }),
        html.P("$10M Mock Portfolio · FRTB-Compliant · 10-Year Historical Simulation",
               style={"color": C["muted"], "fontSize": "0.72rem",
                      "fontFamily": FONT, "margin": "4px 0 0 0"}),
    ], style={"padding": "24px 0 16px", "borderBottom": f"1px solid {C['border']}",
              "marginBottom": "20px"}),

    # ── KPI Row ──
    dbc.Row([
        kpi("Portfolio NAV",   "$10,000,000",   C["accent1"]),
        kpi("VaR 99%",         f"${snap.var_99:,.0f}",  C["accent3"],
            "250-day HistSim"),
        kpi("ES 97.5% (FRTB)", f"${snap.es_frtb:,.0f}", C["accent2"],
            "Liquidity-Horizon Adj."),
        kpi("Div. Benefit",    f"{snap.diversification_benefit:.1%}", C["green"],
            "Cross-asset offset"),
    ], className="g-3 mb-3"),

    dbc.Row([
        kpi("Kupiec Zone",    kupiec["zone"], zone_color.get(kupiec["zone"], C["text"]),
            f"{kupiec['n_exceptions']} exceptions / {kupiec['T']} days"),
        kpi("Exception Rate", f"{kupiec['exception_rate_pct']}%", C["muted"],
            f"Expected: {kupiec['expected_rate_pct']}%"),
        kpi("Christoffersen", "Clustering" if chris["clustering"] else "Independent",
            C["accent2"] if chris["clustering"] else C["green"],
            f"p={chris['p_value']}"),
        kpi("PLA Pass Rate",  f"{pla['pla_pass'].mean():.0%}", C["green"],
            "Spearman ≥0.80 & VarRatio 0.8–1.2"),
    ], className="g-3 mb-4"),

    # ── Tabs ──
    dcc.Tabs(id="tabs", value="var", children=[
        dcc.Tab(label="VaR & ES",         value="var"),
        dcc.Tab(label="P&L Distribution", value="hist"),
        dcc.Tab(label="Asset Attribution",value="attrib"),
        dcc.Tab(label="Greeks",           value="greeks"),
        dcc.Tab(label="Stress Testing",   value="stress"),
        dcc.Tab(label="Backtesting",      value="backtest"),
        dcc.Tab(label="PLA Validation",   value="pla"),
    ], style={"fontFamily": FONT, "fontSize": "0.78rem"}),

    html.Div(id="tab-content", style={"marginTop": "16px"}),

    # ── Stress Table ──
    html.Div([
        html.H5("Limit Breach Monitor",
                style={"fontFamily": FONT, "color": C["accent1"],
                       "fontSize": "0.85rem", "letterSpacing": "0.1em",
                       "marginBottom": "10px"}),
        dash_table.DataTable(
            data=stress_table_data(),
            columns=[{"name": c, "id": c} for c in stress_table_data()[0].keys()],
            style_table={"overflowX": "auto"},
            style_cell={
                "backgroundColor": C["panel"], "color": C["text"],
                "fontFamily": FONT, "fontSize": "0.75rem",
                "padding": "8px 12px", "border": f"1px solid {C['border']}",
                "textAlign": "right",
            },
            style_header={
                "backgroundColor": C["bg"], "color": C["accent1"],
                "fontWeight": "500", "letterSpacing": "0.05em",
            },
            style_data_conditional=[
                {"if": {"filter_query": '{Breach} = "⚠ BREACH"'},
                 "color": C["accent2"], "fontWeight": "600"},
                {"if": {"filter_query": '{Breach} = "✓ OK"'},
                 "color": C["green"]},
            ],
        ),
    ], style={"marginTop": "24px", "padding": "16px",
              "background": C["panel"], "border": f"1px solid {C['border']}",
              "borderRadius": "6px"}),

    html.P("FRTB-Compliant Risk Engine · Historical Simulation · Basel III/IV Framework",
           style={"textAlign": "center", "color": C["muted"],
                  "fontSize": "0.65rem", "fontFamily": FONT,
                  "marginTop": "24px", "paddingBottom": "16px"}),

], fluid=True, style={"backgroundColor": C["bg"], "minHeight": "100vh", "padding": "0 24px"})


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    graph_style = {"height": "420px"}
    if   tab == "var":     fig = fig_var_es()
    elif tab == "hist":    fig = fig_pnl_hist()
    elif tab == "attrib":  fig = fig_asset_contrib()
    elif tab == "greeks":  fig = fig_greeks();  graph_style["height"] = "380px"
    elif tab == "stress":  fig = fig_stress()
    elif tab == "backtest":fig = fig_backtest()
    elif tab == "pla":     fig = fig_pla();     graph_style["height"] = "320px"
    else:                  return html.Div("Unknown tab")

    return dcc.Graph(figure=fig, style=graph_style,
                     config={"displayModeBar": False})


if __name__ == "__main__":
    app.run(debug=False, port=8050, host="0.0.0.0")
