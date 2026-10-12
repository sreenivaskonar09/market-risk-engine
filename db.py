"""
db.py – SQLite persistence layer for the risk engine.
Persists daily risk metrics, backtest logs, stress results, and PLA.
"""
import sqlite3, uuid, time
from datetime import date
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent / "risk_engine.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    schema = (Path(__file__).parent / "schema.sql").read_text()
    conn = get_connection()
    try:
        conn.executescript(schema)
        conn.commit()
        print(f"DB initialised at {DB_PATH}")
    except sqlite3.OperationalError as e:
        if "already exists" not in str(e):
            raise
    finally:
        conn.close()


def persist_risk_metrics(df_risk: pd.DataFrame):
    """Write rolling risk metrics to DB."""
    conn = get_connection()
    run_id = str(uuid.uuid4())
    t0 = time.time()
    rows = 0
    for dt, row in df_risk.iterrows():
        for metric, col in [("VaR_99","var_99"),("ES_975","es_975"),("ES_FRTB","es_frtb")]:
            conn.execute("""
                INSERT OR REPLACE INTO risk_metrics
                    (calc_date, metric_type, value, confidence_lvl, lookback_days)
                VALUES (?,?,?,?,?)
            """, (dt.date().isoformat(), metric, float(row[col]),
                  0.99 if metric=="VaR_99" else 0.975, 250))
        rows += 1
    conn.execute("""
        INSERT INTO run_log (run_id, run_type, status, rows_processed, duration_ms)
        VALUES (?,?,?,?,?)
    """, (run_id, "DAILY_RISK", "SUCCESS", rows, int((time.time()-t0)*1000)))
    conn.commit()
    conn.close()
    print(f"Persisted {rows} risk metric rows.")


def persist_stress_results(df_stress: pd.DataFrame, run_date: str = None):
    run_date = run_date or date.today().isoformat()
    conn = get_connection()
    for _, r in df_stress.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO stress_test_results
            (run_date, scenario_name, eq_pnl, fx_pnl, bond_pnl, vega_pnl,
             total_pnl, pct_of_capital, limit_breach)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (run_date, r["scenario"], r["eq_pnl"], r["fx_pnl"],
              r["bond_pnl"], r["vega_pnl"], r["total_pnl"],
              r["pct_of_capital"], bool(r["limit_breach"])))
    conn.commit()
    conn.close()


def persist_backtest(df_roll: pd.DataFrame, kupiec: dict):
    conn = get_connection()
    exc_mask = df_roll["total_pnl"] < df_roll["var_99"]
    rolling_exc = exc_mask.rolling(250).sum()
    for dt, row in df_roll.iterrows():
        raw = rolling_exc.get(dt, 0)
        n = 0 if (raw is None or (isinstance(raw, float) and pd.isna(raw))) else int(raw)
        rate = n / 250
        zone = "GREEN" if rate <= 0.01 else ("YELLOW" if rate <= 0.015 else "RED")
        conn.execute("""
            INSERT OR REPLACE INTO backtest_log
            (calc_date, var_99, actual_pnl, is_exception, rolling_exceptions,
             rolling_window, zone, kupiec_lr, kupiec_pvalue)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (dt.date().isoformat(), float(row["var_99"]), float(row["total_pnl"]),
              bool(exc_mask.get(dt, False)), n, 250, zone,
              kupiec.get("lr_stat"), kupiec.get("p_value")))
    conn.commit()
    conn.close()


def persist_pla(df_pla: pd.DataFrame):
    conn = get_connection()
    for dt, row in df_pla.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO pla_results
            (calc_date, spearman_r, var_ratio, pla_pass)
            VALUES (?,?,?,?)
        """, (dt.date().isoformat(), float(row["spearman_r"]),
              float(row["var_ratio"]), bool(row["pla_pass"])))
    conn.commit()
    conn.close()


def query_risk_summary(days: int = 30) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(f"""
        SELECT
            r.calc_date,
            MAX(CASE WHEN r.metric_type='VaR_99'  THEN r.value END) AS var_99,
            MAX(CASE WHEN r.metric_type='ES_975'  THEN r.value END) AS es_975,
            MAX(CASE WHEN r.metric_type='ES_FRTB' THEN r.value END) AS es_frtb,
            b.zone,
            b.rolling_exceptions
        FROM risk_metrics r
        LEFT JOIN backtest_log b ON b.calc_date = r.calc_date
        WHERE r.asset_class IS NULL
          AND r.calc_date >= DATE('now', '-{days} days')
        GROUP BY r.calc_date
        ORDER BY r.calc_date DESC
    """, conn)
    conn.close()
    return df


if __name__ == "__main__":
    from data_generator import build_portfolio
    from risk_engine import rolling_risk, run_stress_test, pla_test, kupiec_pof_test

    init_db()
    df     = build_portfolio()
    roll   = rolling_risk(df)
    stress = run_stress_test(df)
    pla    = pla_test(df)
    kupiec = kupiec_pof_test(roll["total_pnl"], roll["var_99"])

    persist_risk_metrics(roll)
    persist_stress_results(stress)
    persist_backtest(roll, kupiec)
    persist_pla(pla)

    print("\nDB summary (last 5 rows):")
    print(query_risk_summary(3000).head())
