"""Run the mechanical TJR backtest on StockPredictor bar data and emit a report + chart.

Tests opposing-liquidity targets AND fixed-R multiples (1R/1.5R/2R/3R) for each market, over the
~2 years of 5-minute history in ``data/bars.sqlite`` -- a far larger sample than the original
60-day yfinance run. Outputs land in ``data/tjr/``.

Run:  python -m stockpredictor.tjr.run
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .backtest import run_backtest  # noqa: E402
from .data import get_daily, get_intraday  # noqa: E402
from .metrics import buy_and_hold_return_pct, compute_stats  # noqa: E402
from .strategy import StrategyConfig  # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "data" / "tjr"
OUT.mkdir(parents=True, exist_ok=True)

# (store ticker, one-way cost per unit, display label)
MARKETS = [("QQQ", 0.01, "Nasdaq ETF"), ("C:EURUSD", 0.00005, "EUR/USD")]
RR_SETTINGS = [1.0, 1.5, 2.0, 3.0]


def run_all():
    report_rows, bh, curves = [], {}, {}
    for tk, cost, label in MARKETS:
        intr = get_intraday(tk)
        daily = get_daily(tk)
        bh[tk] = buy_and_hold_return_pct(intr)
        for rr in RR_SETTINGS:
            res = run_backtest(intr, daily, StrategyConfig(target_mode="fixed_rr", rr=rr), cost)
            s = compute_stats(res.trades, res.start_equity, res.equity_curve)
            report_rows.append((tk, f"{rr:g}R", s))
            curves[(tk, f"{rr:g}R")] = res.equity_curve
        res = run_backtest(intr, daily, StrategyConfig(target_mode="liquidity"), cost)
        s = compute_stats(res.trades, res.start_equity, res.equity_curve)
        report_rows.append((tk, "liquidity", s))
        curves[(tk, "liquidity")] = res.equity_curve
    return report_rows, bh, curves


def make_chart(curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (tk, _cost, label) in zip(axes, MARKETS):
        for setting in ["1R", "1.5R", "2R", "3R", "liquidity"]:
            c = curves.get((tk, setting))
            if c is None or len(c) == 0:
                continue
            series = [10_000.0] + list(c.to_numpy())
            ax.plot(range(len(series)), series, lw=1.2, label=setting)
        ax.axhline(10_000, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{label} ({tk}) — equity by target setting")
        ax.set_xlabel("trade #")
        ax.set_ylabel("equity ($)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"chart -> {path}")


def write_report(report_rows, bh, chart_name, path):
    best = {}
    for tk, _c, label in MARKETS:
        rows_tk = [(setting, s) for r_tk, setting, s in report_rows
                   if r_tk == tk and s.get("trades")]
        if rows_tk:
            best[tk] = max(rows_tk, key=lambda x: x[1]["expectancy_r"])

    L = []
    L.append("# Is TJR's strategy actually profitable? — StockPredictor bar data\n")
    L.append("_Data: `data/bars.sqlite` 5-minute bars, ~2 years. Starting equity $10,000, "
             "risk 1%/trade, costs applied._\n")
    L.append("## Verdict at a glance\n")
    for tk, _c, label in MARKETS:
        if tk not in best:
            L.append(f"- **{label} ({tk})**: no qualifying trades.")
            continue
        setting, s = best[tk]
        verdict = "positive" if s["expectancy_r"] > 0 else "negative"
        L.append(f"- **{label} ({tk})**: best setting **{setting}** at **{s['expectancy_r']:+.2f}R** "
                 f"expectancy over **{s['trades']} trades** ({verdict}); buy & hold "
                 f"**{bh[tk]:+.1f}%**.")
    L.append("")
    L.append("## Full results\n")
    L.append("Columns: **trades | win% | expectancy(R) | profit factor | total return | "
             "max drawdown | tgt/stop/time exits**\n")
    for tk, _c, label in MARKETS:
        L.append(f"### {label} ({tk}) — buy & hold: {bh[tk]:+.1f}%\n")
        L.append("| Target | Trades | Win% | Expectancy | PF | Total return | Max DD | tgt/stop/time |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r_tk, setting, s in report_rows:
            if r_tk != tk:
                continue
            if not s.get("trades"):
                L.append(f"| {setting} | 0 | — | — | — | — | — | — |")
                continue
            pf = s["profit_factor"]
            pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
            L.append(f"| {setting} | {s['trades']} | {s['win_rate']*100:.0f}% | "
                     f"{s['expectancy_r']:+.2f}R | {pf_s} | {s['total_return_pct']:+.1f}% | "
                     f"{s['max_drawdown_pct']:.1f}% | "
                     f"{s['target_hits']}/{s['stop_hits']}/{s['time_exits']} |")
        L.append("")
    L.append(f"## Equity curves\n\n![equity curves]({chart_name})\n")
    L.append("## Caveats\n")
    L.append("1. **Mechanical ≠ discretionary.** Codifies TJR's rules objectively; omits HTF bias, "
             "context, and news filtering a live trader uses.\n"
             "2. **QQQ is the ETF**, not NQ futures (no overnight session in the equity bars); "
             "`C:EURUSD` is Polygon spot forex.\n"
             "3. **Conservative fills**: when a bar spans both stop and target, stop is assumed first.\n"
             "4. Now on ~2 years of data, the trade counts are large enough to be **indicative**; "
             "still a single historical window, not walk-forward validated.\n")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"report -> {path}")


def main():
    rows, bh, curves = run_all()
    make_chart(curves, OUT / "equity_curves.png")
    write_report(rows, bh, "equity_curves.png", OUT / "REPORT.md")
    print("\n=== SUMMARY ===")
    for tk, setting, s in rows:
        if not s.get("trades"):
            print(f"{tk:10} {setting:10} no trades")
            continue
        pf = s["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"{tk:10} {setting:10} n={s['trades']:>3} win={s['win_rate']*100:.0f}% "
              f"exp={s['expectancy_r']:+.2f}R PF={pf_s} ret={s['total_return_pct']:+.1f}% "
              f"tgt/stop/time={s['target_hits']}/{s['stop_hits']}/{s['time_exits']}")
    for tk in bh:
        print(f"buy&hold {tk}: {bh[tk]:+.1f}%")


if __name__ == "__main__":
    main()
