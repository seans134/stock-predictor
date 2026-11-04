import pytest

from stockpredictor.research.event_audit import audit, sample, screen


def test_guidance_is_not_hidden_by_earnings_or_automatically_verified():
    result = screen("Company raises annual guidance after earnings beat", "", ["ABC"])
    assert result["earnings_mentioned"] and result["guidance_mentioned"]
    assert result["upward_candidate"]
    assert result["legacy_category"] == "earnings"
    assert not result["strategy_eligible"] and not result["issuer_verified"]


def test_analyst_forecast_and_multiple_issuers_require_verification():
    result = screen("Analyst raises forecast", "", ["AAPL", "MSFT"])
    assert result["analyst_language"] and result["multiple_tickers"]
    assert not result["strategy_eligible"]
    assert not screen("Company cuts guidance", "", ["AAPL"])["upward_candidate"]


def test_sample_deterministic_stratified_and_order_independent():
    rows = [dict(article_id=str(i), upward_candidate=i%2==0) for i in range(100)]
    chosen = sample(rows)
    assert chosen == sample(list(reversed(rows)))
    assert len(chosen)==20 and sum(r["upward_candidate"] for r in chosen)==12


def test_no_protected_data_access(tmp_path):
    from stockpredictor.research.governance import write_new
    write_new(tmp_path / "data/edge_research/split_manifest.json",
              dict(development_start="2025-01-01", development_end="2025-12-31"))
    with pytest.raises(ValueError, match="protected"):
        audit(tmp_path, tmp_path / "out", "2026-01-01", "2026-01-02")
    assert not (tmp_path / "out").exists()
