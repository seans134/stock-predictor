import datetime as dt

import pandas as pd
import pytest

from stockpredictor.execution.portfolio import RiskConfig, run_portfolio, size_position

DAY = dt.date(2024, 3, 18)
ET = "America/New_York"


def _trade(hhmm, net_ret, q10=-0.005, day=DAY, ticker="AAA"):
    hour, minute = map(int, hhmm.split(":"))
    return {
        "ticker": ticker,
        "date": day,
        "bar_start": pd.Timestamp(day.year, day.month, day.day, hour, minute, tz=ET),
        "direction": "long",
        "gross_ret": net_ret,
        "net_ret": net_ret,
        "q10": q10,
    }


def test_size_position_risk_based():
    config = RiskConfig()
    # 0.5% of 100k = $500 at risk; |q10| = 1% -> $50,000... capped at 20%.
    assert size_position(100_000, -0.01, config) == pytest.approx(20_000)
    # |q10| = 5%: $500 / 0.05 = $10,000, under the cap.
    assert size_position(100_000, -0.05, config) == pytest.approx(10_000)
    # Tiny |q10| hits the floor rather than exploding.
    huge = size_position(100_000, -0.00001, config)
    assert huge == pytest.approx(20_000)  # floored downside, then capped


def test_portfolio_pnl_and_curve():
    config = RiskConfig(max_positions=10)
    trades = pd.DataFrame([_trade("10:00", 0.01), _trade("11:00", -0.01)])
    result = run_portfolio(trades, config)
    # Both size to the 20% cap of equity at entry time.
    first_pnl = 0.20 * config.initial_equity * 0.01
    equity_after_first = config.initial_equity + first_pnl
    second_pnl = 0.20 * equity_after_first * -0.01
    assert result["final_equity"] == pytest.approx(equity_after_first + second_pnl)
    assert result["n_taken"] == 2
    assert result["max_drawdown"] < 0


def test_concurrency_cap_blocks_overlapping_entries():
    config = RiskConfig(max_positions=1)
    # Second trade starts before the first exits (15-min horizon).
    trades = pd.DataFrame([_trade("10:00", 0.01), _trade("10:05", 0.01, ticker="BBB")])
    result = run_portfolio(trades, config)
    assert result["n_taken"] == 1
    assert result["n_skipped_concurrency"] == 1
    # With the positions freed, a later trade is allowed again.
    trades = pd.DataFrame([_trade("10:00", 0.01), _trade("11:00", 0.01, ticker="BBB")])
    assert run_portfolio(trades, config)["n_taken"] == 2


def test_daily_loss_limit_halts_entries():
    config = RiskConfig(max_positions=10, daily_loss_limit=0.001)
    # First trade loses 1% on a 20%-of-equity position = -0.2% of equity,
    # breaching the 0.1% daily limit; later same-day entries are blocked.
    trades = pd.DataFrame(
        [
            _trade("10:00", -0.01),
            _trade("11:00", 0.05),
            _trade("11:30", 0.05),
        ]
    )
    result = run_portfolio(trades, config)
    assert result["n_taken"] == 1
    assert result["n_skipped_daily_halt"] == 2

    # A new day resets the halt.
    next_day = dt.date(2024, 3, 19)
    trades = pd.DataFrame(
        [
            _trade("10:00", -0.01),
            _trade("11:00", 0.05),
            _trade("10:00", 0.02, day=next_day),
        ]
    )
    result = run_portfolio(trades, config)
    assert result["n_taken"] == 2
    assert result["n_skipped_daily_halt"] == 1


def test_empty_trades():
    result = run_portfolio(pd.DataFrame(), RiskConfig())
    assert result["final_equity"] == RiskConfig().initial_equity
    assert result["n_taken"] == 0
