"""Bounded aggregate re-fetch to a separate evidence file; never writes BarStore.

Uses the existing provider credential without logging it. Newly fetched bars
carry observation time only; this is not proof of historical availability.
"""
import argparse
import datetime as dt
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import requests

from stockpredictor.research.baselines import INSTRUMENT_TYPES
from stockpredictor.research.governance import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", choices=sorted(INSTRUMENT_TYPES))
    parser.add_argument("day", type=dt.date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trade-window", type=dt.time.fromisoformat,
                        help="Inspect trades for five minutes starting at this Eastern HH:MM")
    parser.add_argument("--minute-bars", action="store_true", help="Inspect one-minute aggregates instead of five-minute bars")
    args = parser.parse_args()
    if args.trade_window and args.minute_bars:
        parser.error("Choose trades or minute bars, not both")
    if args.output.exists():
        raise FileExistsError("Evidence file already exists")
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    key = os.getenv("POLYGON_API_KEY")
    result = dict(ticker=args.ticker, date=str(args.day), adjusted=False,
                  observed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  historical_availability="unverified", database_modified=False)
    result["aggregate_minutes"] = 1 if args.minute_bars else 5
    if not key:
        result["status"] = "credential_not_configured"
    else:
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{args.ticker}/range/{result['aggregate_minutes']}/minute/{args.day}/{args.day}"
            params = {"adjusted": "false", "sort": "asc", "limit": 50000}
            if args.trade_window:
                start = dt.datetime.combine(args.day, args.trade_window, tzinfo=ZoneInfo("America/New_York"))
                url = f"https://api.polygon.io/v3/trades/{args.ticker}"
                params = {"timestamp.gte": int(start.timestamp())*1_000_000_000,
                          "timestamp.lt": int((start+dt.timedelta(minutes=5)).timestamp())*1_000_000_000,
                          "sort": "timestamp", "order": "asc", "limit": 50000}
                result["trade_window_start"] = start.isoformat()
            response = requests.get(
                url, params=params,
                headers={"Authorization": "Bearer " + key}, timeout=30)
            result["http_status"] = response.status_code
            if response.status_code == 200:
                payload = response.json()
                result.update(status=payload.get("status"), has_more=bool(payload.get("next_url")))
                result["trades" if args.trade_window else "bars"] = payload.get("results", [])
            else:
                result["status"] = "provider_request_rejected"
        except (requests.RequestException, ValueError) as exc:
            # Exception strings/URLs may carry credentials; retain type only.
            result.update(status="network_or_payload_error", error_type=type(exc).__name__)
    write_new(args.output, result)
    print({k: v for k, v in result.items() if k not in ("bars", "trades")})
    print("returned_bars", len(result.get("bars") or []))
    if args.trade_window:
        print("returned_trades", len(result.get("trades") or []))


if __name__ == "__main__":
    main()
