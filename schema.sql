-- ============================================================
-- Multi-Asset Risk Engine – Database Schema
-- Compatible with: SQLite, PostgreSQL, SQL Server
-- ============================================================

-- ── Market Data ──────────────────────────────────────────────
CREATE TABLE market_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      DATE        NOT NULL,
    asset_class     VARCHAR(20) NOT NULL,  -- 'EQUITY', 'FX', 'BOND'
    ticker          VARCHAR(20) NOT NULL,
    price           DECIMAL(18,6) NOT NULL,
    daily_return    DECIMAL(12,8),
    impl_vol        DECIMAL(8,4),          -- for options
    yield           DECIMAL(8,6),          -- for bonds
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, ticker)
);

-- ── Portfolio Positions ───────────────────────────────────────
CREATE TABLE portfolio_positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      DATE        NOT NULL,
    instrument_id   VARCHAR(50) NOT NULL,
    asset_class     VARCHAR(20) NOT NULL,
    notional        DECIMAL(18,2) NOT NULL,
    delta           DECIMAL(10,6),
    gamma           DECIMAL(14,10),
    vega            DECIMAL(12,6),
    daily_pnl       DECIMAL(18,2),
    UNIQUE(trade_date, instrument_id)
);

-- ── Risk Metrics (Rolling) ────────────────────────────────────
CREATE TABLE risk_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    calc_date       DATE        NOT NULL,
    metric_type     VARCHAR(30) NOT NULL,  -- 'VaR_99', 'ES_975', 'ES_FRTB'
    asset_class     VARCHAR(20),           -- NULL = portfolio level
    value           DECIMAL(18,2) NOT NULL,
    confidence_lvl  DECIMAL(5,4),
    lookback_days   INTEGER,
    lh_days         INTEGER,               -- liquidity horizon (FRTB)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(calc_date, metric_type, asset_class)
);

-- ── Stress Test Results ───────────────────────────────────────
CREATE TABLE stress_test_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        DATE        NOT NULL,
    scenario_name   VARCHAR(60) NOT NULL,
    eq_pnl          DECIMAL(18,2),
    fx_pnl          DECIMAL(18,2),
    bond_pnl        DECIMAL(18,2),
    vega_pnl        DECIMAL(18,2),
    total_pnl       DECIMAL(18,2),
    pct_of_capital  DECIMAL(8,4),
    limit_breach    BOOLEAN     NOT NULL DEFAULT FALSE,
    capital_limit   DECIMAL(18,2),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Backtest Log ──────────────────────────────────────────────
CREATE TABLE backtest_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    calc_date       DATE        NOT NULL,
    var_99          DECIMAL(18,2),
    actual_pnl      DECIMAL(18,2),
    is_exception    BOOLEAN,
    rolling_exceptions INTEGER,
    rolling_window  INTEGER,
    zone            VARCHAR(10),           -- GREEN / YELLOW / RED
    kupiec_lr       DECIMAL(10,4),
    kupiec_pvalue   DECIMAL(8,4),
    chris_lr        DECIMAL(10,4),
    chris_pvalue    DECIMAL(8,4),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(calc_date)
);

-- ── PLA Results ───────────────────────────────────────────────
CREATE TABLE pla_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    calc_date       DATE        NOT NULL,
    spearman_r      DECIMAL(8,4),
    var_ratio       DECIMAL(8,4),
    pla_pass        BOOLEAN,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(calc_date)
);

-- ── Audit / Run Log ───────────────────────────────────────────
CREATE TABLE run_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          VARCHAR(36) NOT NULL,  -- UUID
    run_type        VARCHAR(30),           -- 'DAILY_RISK', 'STRESS', 'BACKTEST'
    status          VARCHAR(10),           -- 'SUCCESS', 'FAILED'
    rows_processed  INTEGER,
    duration_ms     INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Indexes ───────────────────────────────────────────────────
CREATE INDEX idx_market_data_date   ON market_data(trade_date);
CREATE INDEX idx_risk_metrics_date  ON risk_metrics(calc_date, metric_type);
CREATE INDEX idx_stress_run_date    ON stress_test_results(run_date);
CREATE INDEX idx_backtest_date      ON backtest_log(calc_date);

-- ── Sample Query: Daily Risk Report ──────────────────────────
-- SELECT
--     r.calc_date,
--     MAX(CASE WHEN r.metric_type = 'VaR_99'   THEN r.value END) AS var_99,
--     MAX(CASE WHEN r.metric_type = 'ES_975'   THEN r.value END) AS es_975,
--     MAX(CASE WHEN r.metric_type = 'ES_FRTB'  THEN r.value END) AS es_frtb,
--     b.is_exception,
--     b.zone
-- FROM risk_metrics r
-- LEFT JOIN backtest_log b ON b.calc_date = r.calc_date
-- WHERE r.asset_class IS NULL
--   AND r.calc_date >= DATE('now', '-30 days')
-- GROUP BY r.calc_date, b.is_exception, b.zone
-- ORDER BY r.calc_date DESC;
