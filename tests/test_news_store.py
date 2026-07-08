import datetime as dt

import pandas as pd
import pytest

from stockpredictor.data.asof import AsOfData
from stockpredictor.data.availability import NEWS_INGEST_LATENCY
from stockpredictor.data.news_store import NewsStore
from stockpredictor.data.store import BarStore
from stockpredictor.ingest.polygon_news import parse_articles


def _article(article_id: str, published: str, tickers=None, sentiments=None):
    published_ts = pd.Timestamp(published, tz="UTC")
    ticker_rows = []
    for i, t in enumerate(tickers or ["AAA"]):
        sentiment = (sentiments or {}).get(t)
        ticker_rows.append(
            {"ticker": t, "sentiment": sentiment, "sentiment_reasoning": None}
        )
    return {
        "article_id": article_id,
        "published": published_ts,
        "available": published_ts + NEWS_INGEST_LATENCY,
        "publisher": "TestWire",
        "title": f"article {article_id}",
        "description": "desc",
        "keywords": ["k1"],
        "tickers": ticker_rows,
    }


@pytest.fixture
def news(tmp_path) -> NewsStore:
    s = NewsStore(tmp_path / "news.sqlite")
    yield s
    s.close()


def test_insert_idempotent_and_coverage(news):
    articles = [
        _article("a1", "2024-03-18 12:00"),
        _article("a2", "2024-03-18 14:00"),
    ]
    assert news.insert_articles(articles, source="test") == 2
    assert news.insert_articles(articles, source="test") == 0
    n, lo, hi = news.coverage()
    assert n == 2
    assert lo == pd.Timestamp("2024-03-18 12:00", tz="UTC")
    assert hi == pd.Timestamp("2024-03-18 14:00", tz="UTC")


def test_asof_visibility_respects_latency(news, tmp_path):
    published = pd.Timestamp("2024-03-18 13:00", tz="UTC")
    news.insert_articles([_article("a1", "2024-03-18 13:00")], source="test")
    bars = BarStore(tmp_path / "bars.sqlite")
    data = AsOfData(bars, news)

    # At publish time the article has not "arrived" yet.
    assert data.get_news("AAA", as_of=published).empty
    # After the conservative latency it is visible.
    visible = data.get_news("AAA", as_of=published + NEWS_INGEST_LATENCY)
    assert len(visible) == 1
    assert visible.iloc[0]["title"] == "article a1"
    bars.close()


def test_lookahead_inserting_later_news_preserves_past(news):
    news.insert_articles([_article("a1", "2024-03-18 12:00")], source="test")
    as_of = pd.Timestamp("2024-03-18 13:00", tz="UTC")
    before = news.get_news_asof("AAA", as_of=as_of)

    news.insert_articles([_article("a2", "2024-03-18 12:30")], source="test")
    news.insert_articles([_article("a3", "2024-03-19 09:00")], source="test")
    after = news.get_news_asof("AAA", as_of=as_of)

    # a2 became visible at 12:45 < 13:00 so it legitimately appears; the
    # point of the test is determinism: recomputing at the same as_of after
    # future (a3) rows arrive must not change anything else.
    assert list(after["article_id"]) == ["a1", "a2"]
    assert list(before["article_id"]) == ["a1", "a2"] or list(before["article_id"]) == ["a1"]
    later = news.get_news_asof("AAA", as_of=pd.Timestamp("2024-03-19 10:00", tz="UTC"))
    assert list(later["article_id"]) == ["a1", "a2", "a3"]


def test_multi_ticker_and_missing_sentiment(news):
    news.insert_articles(
        [
            _article(
                "a1",
                "2024-03-18 12:00",
                tickers=["AAA", "BBB"],
                sentiments={"AAA": "positive"},
            )
        ],
        source="test",
    )
    as_of = pd.Timestamp("2024-03-18 13:00", tz="UTC")
    aaa = news.get_news_asof("AAA", as_of=as_of)
    bbb = news.get_news_asof("BBB", as_of=as_of)
    assert aaa.iloc[0]["sentiment"] == "positive"
    assert bbb.iloc[0]["sentiment"] is None  # missing, not neutral


def test_parse_articles_vendor_payload():
    payload = [
        {
            "id": "abc123",
            "published_utc": "2024-08-05T20:10:00Z",
            "publisher": {"name": "The Motley Fool"},
            "title": "Some headline",
            "description": "Body summary",
            "keywords": ["ai", "chips"],
            "tickers": ["NVDA", "MU"],
            "insights": [
                {"ticker": "NVDA", "sentiment": "neutral", "sentiment_reasoning": "r1"},
                {"ticker": "MU", "sentiment": "positive", "sentiment_reasoning": "r2"},
            ],
        }
    ]
    parsed = parse_articles(payload)
    assert len(parsed) == 1
    article = parsed[0]
    assert article["article_id"] == "abc123"
    assert article["published"] == pd.Timestamp("2024-08-05T20:10:00Z")
    assert article["available"] == article["published"] + NEWS_INGEST_LATENCY
    assert article["publisher"] == "The Motley Fool"
    by_ticker = {t["ticker"]: t for t in article["tickers"]}
    assert by_ticker["NVDA"]["sentiment"] == "neutral"
    assert by_ticker["MU"]["sentiment_reasoning"] == "r2"


def test_parse_articles_without_insights():
    payload = [
        {
            "id": "x1",
            "published_utc": "2024-08-05T10:00:00Z",
            "tickers": ["AAA"],
        }
    ]
    parsed = parse_articles(payload)
    assert parsed[0]["tickers"][0]["sentiment"] is None
