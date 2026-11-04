"""Outcome-blind H2 news audit. Keyword candidates are never verified events."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sqlite3

from stockpredictor.news.events import classify_event
from stockpredictor.research.governance import digest, fingerprint, require_development, write_new

RULES = {
    "version": "h2-event-audit-v1", "performance_access": "prohibited",
    "candidate": "guidance/outlook/forecast language and upward revision language; screening only",
    "review_sample": "12 candidate and 8 noncandidate articles; deterministic SHA256 rank; no return-based selection",
    "verification": "primary source URL and archived text; issuer; fiscal period; metric; units; prior and new comparable guidance; publication and actual first-seen evidence; reviewer decision",
    "pass_rule": "zero critical issuer, direction, comparability or timestamp errors in reviewed accepted events; no unreviewed event accepted",
    "limitations": "Small stratified quality sample is not a population accuracy estimate or sufficient statistical power for a strategy test.",
}
GUIDANCE = re.compile(r"\b(guidance|outlook|forecast)\b", re.I)
UP = re.compile(r"\b(raises?|raised|raising|lifts?|lifted|boosts?|boosted|increases?|increased|upward)\b", re.I)
EARNINGS = re.compile(r"\b(earnings|eps|quarterly results)\b", re.I)
ANALYST = re.compile(r"\b(analysts?|price targets?|upgrades?)\b", re.I)


def screen(title, description, tickers):
    text = f"{title or ''} {description or ''}"
    guidance = bool(GUIDANCE.search(text))
    candidate = guidance and bool(UP.search(text))
    return dict(guidance_mentioned=guidance, earnings_mentioned=bool(EARNINGS.search(text)),
                upward_candidate=candidate, analyst_language=bool(ANALYST.search(text)),
                multiple_tickers=len(tickers) != 1, issuer_verified=False,
                legacy_category=classify_event(title, description),
                status="needs_source_verification" if candidate else "not_screened_in",
                strategy_eligible=False)


def sample(rows):
    ranked = sorted(rows, key=lambda r: fingerprint([RULES["version"], r["article_id"]]))
    return ([r for r in ranked if r["upward_candidate"]][:12]
            + [r for r in ranked if not r["upward_candidate"]][:8])


def audit(root, output, start, end):
    manifest = json.loads((root / "data/edge_research/split_manifest.json").read_text())
    require_development(manifest, start, end)
    db = root / "data/news.sqlite"
    before = digest(db)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "rules.json", dict(rules=RULES, start=start, end=end, database_sha256=before,
        source_sha256=digest(Path(__file__)), created_at=dt.datetime.now(dt.timezone.utc).isoformat()))
    # Dates are UTC publication dates, explicitly separate from exchange session dates.
    lo = int(dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp())
    hi = int((dt.datetime.fromisoformat(end)+dt.timedelta(days=1)).replace(tzinfo=dt.timezone.utc).timestamp())
    with sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("""SELECT a.*,
            (SELECT GROUP_CONCAT(ticker, ',') FROM article_tickers t WHERE t.article_id=a.article_id) tagged
            FROM articles a WHERE published_ts>=? AND published_ts<? ORDER BY published_ts, article_id""", (lo, hi))]
    for row in rows:
        row["tickers"] = sorted((row.pop("tagged") or "").split(",")) if row.get("tagged") else []
        row.pop("tagged", None)
        row.update(screen(row["title"], row["description"], row["tickers"]))
        row["publication_to_availability_seconds"] = row["available_ts"]-row["published_ts"]
    selected = sample(rows)
    review = []
    for row in selected:
        review.append({**row, "review": dict(primary_source_url=None, archived_source_path=None,
            issuer=None, fiscal_period=None, metric=None, units=None, prior_guidance=None,
            new_guidance=None, comparable=None, first_seen_at=None, processing_completed_at=None,
            event_cluster_id=None, reviewer_decision="pending", exclusion_reason=None)})
    with (output / "candidates.jsonl").open("x", encoding="utf-8") as out:
        for row in rows:
            if row["upward_candidate"]:
                out.write(json.dumps(row)+"\n")
    write_new(output / "review_sample.json", review)
    if digest(db) != before:
        raise ValueError("News database changed during audit")
    summary = dict(articles=len(rows), upward_candidates=sum(r["upward_candidate"] for r in rows),
        guidance_mentions=sum(r["guidance_mentioned"] for r in rows),
        guidance_hidden_by_earnings_priority=sum(r["guidance_mentioned"] and r["legacy_category"]=="earnings" for r in rows),
        candidate_multiple_tickers=sum(r["upward_candidate"] and r["multiple_tickers"] for r in rows),
        candidate_analyst_language=sum(r["upward_candidate"] and r["analyst_language"] for r in rows),
        fixed_900_second_availability=sum(r["publication_to_availability_seconds"]==900 for r in rows),
        reviewed_sample_size=len(review), verified_events=0, strategy_eligible_events=0,
        status="blocked_pending_primary_source_and_timestamp_verification", database_sha256=before,
        missing_schema_fields=["primary_source_url", "full_source_text", "source_versions", "actual_first_seen_at", "processing_completed_at", "comparable_prior_guidance"])
    write_new(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output, args.start, args.end)


if __name__ == "__main__":
    main()
