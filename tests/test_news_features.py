import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.data.availability import NEWS_INGEST_LATENCY
from stockpredictor.data.news_store import NewsStore
from stockpredictor.news.clustering import cluster_articles, normalize_title
from stockpredictor.news.features import (
    STAGE1_NEWS_FEATURES,
    STAGE2_NEWS_FEATURES,
    NewsFeatureBuilder,
)


def test_normalize_title():
    assert normalize_title("NVIDIA Soars 10%: What's Next?") == "nvidia soars 10 what s next"
    assert normalize_title(None) == ""


def _articles_frame(rows):
    return pd.DataFrame(
        [
            {
                "article_id": r[0],
                "published_ts": pd.Timestamp(r[1], tz="UTC"),
                "available_ts": pd.Timestamp(r[1], tz="UTC") + NEWS_INGEST_LATENCY,
                "publisher": r[2],
                "title": r[3],
            }
            for r in rows
        ]
    )


def test_syndicated_copies_share_a_cluster():
    frame = _articles_frame(
        [
            ("a1", "2024-03-18 12:00", "WireA", "Acme Corp Beats Earnings Estimates"),
            ("a2", "2024-03-18 14:00", "WireB", "Acme Corp beats earnings estimates!"),
            ("a3", "2024-03-18 15:00", "WireC", "Completely different story"),
            # Same title but far outside the 72h window: new cluster.
            ("a4", "2024-03-25 12:00", "WireA", "Acme Corp Beats Earnings Estimates"),
        ]
    )
    clustered = cluster_articles(frame).set_index("article_id")
    assert clustered.loc["a2", "cluster_id"] == clustered.loc["a1", "cluster_id"]
    assert clustered.loc["a3", "cluster_id"] != clustered.loc["a1", "cluster_id"]
    assert clustered.loc["a4", "cluster_id"] != clustered.loc["a1", "cluster_id"]
    assert clustered.loc["a1", "is_cluster_start"]
    assert not clustered.loc["a2", "is_cluster_start"]


def test_clustering_is_point_in_time():
    """Appending later articles never changes an earlier assignment."""
    early = _articles_frame(
        [
            ("a1", "2024-03-18 12:00", "WireA", "Acme Corp Beats Earnings"),
            ("a2", "2024-03-18 14:00", "WireB", "Acme Corp Beats Earnings"),
        ]
    )
    late = _articles_frame([("a3", "2024-03-19 09:00", "WireC", "Acme Corp Beats Earnings")])

    first = cluster_articles(early).set_index("article_id")
    combined = cluster_articles(pd.concat([early, late])).set_index("article_id")
    for aid in ("a1", "a2"):
        assert combined.loc[aid, "cluster_id"] == first.loc[aid, "cluster_id"]
        assert combined.loc[aid, "is_cluster_start"] == first.loc[aid, "is_cluster_start"]


@pytest.fixture
def store_with_news(tmp_path):
    store = NewsStore(tmp_path / "news.sqlite")
    published = pd.Timestamp("2024-03-18 08:00", tz="UTC")

    def art(article_id, published, title, publisher, sentiment):
        return {
            "article_id": article_id,
            "published": published,
            "available": published + NEWS_INGEST_LATENCY,
            "publisher": publisher,
            "title": title,
            "description": "",
            "keywords": [],
            "tickers": [
                {"ticker": "AAA", "sentiment": sentiment, "sentiment_reasoning": None}
            ],
        }

    store.insert_articles(
        [
            # Two syndicated copies of one positive event + one distinct event.
            art("n1", published, "Acme wins huge contract", "WireA", "positive"),
            art("n2", published + pd.Timedelta(hours=1), "Acme Wins Huge Contract", "WireB", "positive"),
            art("n3", published + pd.Timedelta(hours=2), "Acme faces lawsuit", "WireC", "negative"),
            # An article published after the cutoff we will query.
            art("n4", published + pd.Timedelta(days=1), "Acme update", "WireA", None),
        ],
        source="test",
    )
    yield store
    store.close()


def test_stage1_news_features_counts_and_cutoff(store_with_news):
    builder = NewsFeatureBuilder(store_with_news)
    cutoff = pd.Timestamp("2024-03-18 14:00", tz="UTC")
    feats = builder.stage1_features("AAA", pd.DatetimeIndex([cutoff]))

    row = feats.iloc[0]
    # Two clusters (syndicated pair counts once), three sources.
    assert row["news_clusters_24h"] == 2.0
    assert row["news_clusters_7d"] == 2.0
    assert row["news_source_diversity"] == 3.0
    # One positive event, one negative: signed roughly cancels, magnitude adds.
    assert abs(row["news_sent_signed"]) < row["news_sent_magnitude"]
    assert row["news_novelty_frac"] == 1.0  # both events first seen today

    # Cutoff before any availability: everything is zero, nothing leaks.
    early = builder.stage1_features(
        "AAA", pd.DatetimeIndex([pd.Timestamp("2024-03-18 07:00", tz="UTC")])
    )
    assert early.iloc[0]["news_clusters_24h"] == 0.0


def test_stage1_features_unknown_ticker_is_zero(store_with_news):
    builder = NewsFeatureBuilder(store_with_news)
    feats = builder.stage1_features(
        "ZZZ", pd.DatetimeIndex([pd.Timestamp("2024-03-18 14:00", tz="UTC")])
    )
    assert feats.iloc[0]["news_clusters_24h"] == 0.0


def test_stage2_news_window_counts(store_with_news):
    builder = NewsFeatureBuilder(store_with_news)
    # n1 available 08:15, n2 09:15, n3 10:15 UTC.
    bar_ends = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-03-18 08:00", tz="UTC"),   # nothing visible yet
            pd.Timestamp("2024-03-18 10:00", tz="UTC"),   # n1, n2 within 2h
            pd.Timestamp("2024-03-18 12:00", tz="UTC"),   # n2, n3 within 2h
        ]
    )
    feats = builder.stage2_features("AAA", bar_ends)
    # At 12:00 the 2h window (10:00, 12:00] holds only n3 (n2 arrived 09:15).
    assert list(feats["news_articles_2h"]) == [0.0, 2.0, 1.0]
    assert list(feats["news_articles_24h"]) == [0.0, 2.0, 3.0]
    assert feats["news_sent_signed_2h"].iloc[2] == pytest.approx(-1.0)  # n3 negative
    assert set(feats.columns) == set(STAGE2_NEWS_FEATURES)


def test_stage1_feature_list_registered():
    from stockpredictor.stage1.features import STAGE1_FEATURES

    for col in STAGE1_NEWS_FEATURES:
        assert col in STAGE1_FEATURES
