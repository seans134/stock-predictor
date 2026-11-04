"""Append-only issuer evidence. Observations never backdate availability."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from stockpredictor.research.governance import fingerprint


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timezone required")
    return parsed.astimezone(dt.timezone.utc)


class EvidenceStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.folder / "evidence.sqlite")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY, url TEXT NOT NULL UNIQUE, issuer TEXT NOT NULL, ticker TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS versions (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
                sha256 TEXT NOT NULL, first_observed_at TEXT NOT NULL, processed_at TEXT NOT NULL,
                UNIQUE(source_id, sha256));
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
                requested_at TEXT NOT NULL, received_at TEXT NOT NULL, completed_at TEXT NOT NULL,
                http_status INTEGER, outcome TEXT NOT NULL, version_id TEXT REFERENCES versions(id));
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL REFERENCES events(id),
                recorded_at TEXT NOT NULL, reviewer TEXT NOT NULL, decision TEXT NOT NULL,
                checks TEXT NOT NULL, note TEXT NOT NULL);
        """)
        # Preserve revisions and failed requests; callers cannot silently replace evidence.
        for table in ("sources", "versions", "observations", "events", "reviews"):
            for action in ("UPDATE", "DELETE"):
                self.conn.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} "
                                  f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END")
        self.conn.commit()

    def close(self):
        self.conn.close()

    def add_source(self, url, issuer, ticker):
        source_id = fingerprint([url, issuer, ticker])
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO sources VALUES(?,?,?,?)", (source_id, url, issuer, ticker))
        if not self.conn.execute("SELECT 1 FROM sources WHERE id=?", (source_id,)).fetchone():
            raise ValueError("URL already registered to a different issuer")
        return source_id

    def observe(self, source_id, requested_at, received_at, http_status, outcome, body=None):
        completed_at = utc_now()
        if not timestamp(requested_at) <= timestamp(received_at) <= timestamp(completed_at):
            raise ValueError("Invalid observation chronology")
        if (outcome == "ok") != (body is not None and http_status == 200):
            raise ValueError("Successful observations require HTTP 200 and archived bytes")
        version_id = None
        with self.conn:
            if outcome == "ok":
                sha = hashlib.sha256(body).hexdigest()
                version_id = fingerprint([source_id, sha])
                path = self.folder / "blobs" / (sha + ".html")
                path.parent.mkdir(exist_ok=True)
                if path.exists():
                    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                        raise ValueError("Archived source was modified")
                else:
                    with path.open("xb") as out:
                        out.write(body)
                self.conn.execute("INSERT OR IGNORE INTO versions VALUES(?,?,?,?,?)",
                                  (version_id, source_id, sha, received_at, completed_at))
            self.conn.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, source_id, requested_at, received_at, completed_at, http_status, outcome, version_id))
        return version_id

    def version(self, version_id):
        row = self.conn.execute("SELECT v.*,s.issuer,s.ticker,s.url FROM versions v "
            "JOIN sources s ON s.id=v.source_id WHERE v.id=?", (version_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown source version")
        result = dict(row)
        if hashlib.sha256((self.folder / "blobs" / (row["sha256"]+".html")).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Archived source was modified")
        return result

    def add_event(self, payload):
        required = ("ticker", "issuer", "event_type", "fiscal_period", "metric", "units", "cluster_id",
                    "prior_version_id", "new_version_id", "prior_range", "new_range", "publication_claim")
        if any(key not in payload or payload[key] is None for key in required):
            raise ValueError("Incomplete event claim")
        if payload["event_type"] not in ("revision", "initiation", "reaffirmation", "withdrawal"):
            raise ValueError("Unknown event type")
        for key in ("prior_range", "new_range"):
            values = payload[key]
            import math
            if len(values) != 2 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values) or values[0] > values[1]:
                raise ValueError("Invalid guidance range")
        for key in ("prior_version_id", "new_version_id"):
            version = self.version(payload[key])
            if (version["ticker"], version["issuer"]) != (payload["ticker"], payload["issuer"]):
                raise ValueError("Issuer does not match archived source")
        identifier = fingerprint(payload)
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO events VALUES(?,?,?)", (identifier, utc_now(), json.dumps(payload, allow_nan=False)))
        return identifier

    def review(self, event_id, reviewer, decision, checks, note):
        if not reviewer.strip() or not note.strip() or decision not in ("pending", "accepted", "rejected"):
            raise ValueError("Reviewer, decision and rationale required")
        required = ("issuer", "direction", "comparability", "prior_is_latest", "publication_time", "novelty", "cluster")
        if decision == "accepted" and any(checks.get(key) is not True for key in required):
            raise ValueError("Accepted review requires all critical checks")
        with self.conn:
            self.conn.execute("INSERT INTO reviews(event_id,recorded_at,reviewer,decision,checks,note) VALUES(?,?,?,?,?,?)",
                              (event_id, utc_now(), reviewer, decision, json.dumps(checks), note))

    def readiness(self, event_id, as_of, max_latency_seconds=900):
        """Evidence gate only, not an order or strategy signal. Uses latest known review."""
        now = timestamp(as_of)
        if max_latency_seconds < 0:
            raise ValueError("Latency bound must be nonnegative")
        row = self.conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown event")
        event = json.loads(row["payload"])
        reasons = []
        if timestamp(row["created_at"]) > now:
            reasons.append("claim_not_available")
        reviews = self.conn.execute("SELECT * FROM reviews WHERE event_id=? ORDER BY id DESC", (event_id,)).fetchall()
        review = next((r for r in reviews if timestamp(r["recorded_at"]) <= now), None)
        if review is None or review["decision"] != "accepted":
            reasons.append("critical_review_incomplete")
        versions = [self.version(event[key]) for key in ("prior_version_id", "new_version_id")]
        if any(timestamp(v["processed_at"]) > now for v in versions):
            reasons.append("source_not_processed")
        claim = event["publication_claim"]
        try:
            published = timestamp(claim)
            observed = timestamp(versions[1]["first_observed_at"])
            if not 0 <= (observed-published).total_seconds() <= max_latency_seconds:
                reasons.append("historical_backfill_or_invalid_latency")
            if not 0 <= (now-published).total_seconds() <= max_latency_seconds:
                reasons.append("event_stale_or_future")
        except (ValueError, TypeError):
            reasons.append("exact_publication_time_unverified")
        if event["event_type"] != "revision" or not all(n > p for n,p in zip(event["new_range"], event["prior_range"])):
            reasons.append("not_unambiguous_upward_revision")
        return dict(event_id=event_id, evidence_ready=not reasons, blockers=reasons,
                    strategy_eligible=False, note="Strategy timing, universe and duplicate-cluster controls remain separate.")
