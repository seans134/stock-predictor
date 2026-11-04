import datetime as dt
import sqlite3

import pytest

from stockpredictor.news import evidence
from stockpredictor.research.collect_evidence import collect

T = "2026-10-21T18:00:00+00:00"
CHECKS = dict.fromkeys(("issuer", "direction", "comparability", "prior_is_latest", "publication_time", "novelty", "cluster"), True)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "utc_now", lambda: T)
    value = evidence.EvidenceStore(tmp_path)
    yield value
    value.close()


def version(store, url="https://issuer.example/release", body=b"source"):
    sid = store.add_source(url, "Issuer", "DD")
    vid = store.observe(sid, T, T, 200, "ok", body)
    return sid, vid


def event(store, publication="2026-10-21T17:59:00+00:00"):
    _, prior = version(store, "https://issuer.example/prior", b"prior")
    _, new = version(store, body=b"new")
    payload = dict(ticker="DD", issuer="Issuer", event_type="revision", fiscal_period="FY2026",
                   metric="adjusted EPS", units="USD/share", cluster_id="draft-cluster", prior_version_id=prior,
                   new_version_id=new, prior_range=[2.25, 2.30], new_range=[2.35, 2.40], publication_claim=publication)
    return store.add_event(payload)


def test_unchanged_observation_preserves_first_seen_and_changed_page_adds_version(store, monkeypatch):
    sid, vid = version(store)
    later = "2026-10-21T18:01:00+00:00"
    monkeypatch.setattr(evidence, "utc_now", lambda: later)
    assert store.observe(sid, later, later, 200, "ok", b"source") == vid
    other = store.observe(sid, later, later, 200, "ok", b"changed source")
    assert other != vid
    assert store.version(vid)["first_observed_at"] == T
    assert store.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3
    assert store.conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 2


def test_failed_request_cannot_become_an_empty_valid_source(store):
    sid = store.add_source("https://issuer.example/release", "Issuer", "DD")
    assert store.observe(sid, T, T, 403, "http_rejected") is None
    with pytest.raises(ValueError, match="Successful"):
        store.observe(sid, T, T, 403, "ok", b"denied")
    assert store.conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 0


def test_sql_updates_and_deletes_rejected_and_disk_tampering_detected(store):
    _, vid = version(store)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("DELETE FROM versions")
    row = store.version(vid)
    (store.folder / "blobs" / (row["sha256"]+".html")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="modified"):
        store.version(vid)


def test_pending_review_and_future_observation_cannot_qualify(store):
    identifier = event(store)
    assert "critical_review_incomplete" in store.readiness(identifier, T)["blockers"]
    with pytest.raises(ValueError, match="critical checks"):
        store.review(identifier, "reviewer", "accepted", {}, "not verified")
    store.review(identifier, "reviewer", "accepted", CHECKS, "all checks verified for synthetic fixture")
    result = store.readiness(identifier, T)
    assert result["evidence_ready"] and not result["strategy_eligible"]
    earlier = store.readiness(identifier, "2026-10-21T17:59:30+00:00")
    assert not earlier["evidence_ready"] and "source_not_processed" in earlier["blockers"]
    assert not store.readiness(identifier, "2026-10-21T18:30:00+00:00")["evidence_ready"]


@pytest.mark.parametrize("publication", ["2026-05-05", "2026-05-05T10:00:00+00:00"])
def test_historical_or_date_only_publication_is_not_prospective(store, publication):
    identifier = event(store, publication)
    store.review(identifier, "reviewer", "accepted", CHECKS, "synthetic content review")
    assert not store.readiness(identifier, T)["evidence_ready"]


def test_new_rejection_supersedes_old_acceptance_without_erasing_it(store, monkeypatch):
    identifier = event(store)
    store.review(identifier, "reviewer", "accepted", CHECKS, "initial synthetic review")
    later = "2026-10-21T18:01:00+00:00"
    monkeypatch.setattr(evidence, "utc_now", lambda: later)
    store.review(identifier, "reviewer", "rejected", {}, "new contrary evidence")
    assert store.readiness(identifier, T)["evidence_ready"]
    assert not store.readiness(identifier, later)["evidence_ready"]
    assert store.conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 2


def test_collector_bounded_and_rejects_credentials_before_network(store):
    with pytest.raises(ValueError, match="bounded"):
        collect(store, [], rounds=100)
    with pytest.raises(ValueError, match="credential-free"):
        collect(store, [dict(url="https://secret@issuer.example/release", issuer="Issuer", ticker="DD")])


def test_rejects_cross_issuer_source_mapping(store):
    version(store)
    with pytest.raises(ValueError, match="different issuer"):
        store.add_source("https://issuer.example/release", "Someone else", "XYZ")
