"""News feature aggregation for Stage 1 and Stage 2.

All aggregation is over availability timestamps, so a feature computed
"as of T" uses only articles the system could have had at T. Clusters
(not raw articles) are the unit for Stage 1, so one syndicated story never
counts five times; Stage 2 uses cheap vectorized article-window counts.

Sentiment encoding: positive=+1, negative=-1, neutral=0, absent=excluded.
A day with no qualifying news gets zero counts and zero sentiment plus the
count features to disambiguate — no news is never read as negative news.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.data.news_store import NewsStore
from stockpredictor.news.clustering import cluster_articles
from stockpredictor.news.events import FEATURED_EVENT_TYPES, classify_event

# Half-life of an event's influence on the pre-market snapshot. Named
# starting value, tunable per fold like every other constant.
DECAY_HALF_LIFE_HOURS = 24.0
EPS = 1e-9

STAGE1_NEWS_FEATURES = [
    "news_clusters_24h",      # distinct events visible in the last 24h
    "news_clusters_7d",
    "news_decayed_weight",    # sum of exp-decayed event weights over 7d
    "news_sent_signed",       # decay-weighted signed entity sentiment
    "news_sent_magnitude",    # decay-weighted |sentiment|
    "news_source_diversity",  # distinct publishers in the last 24h
    "news_novelty_frac",      # fraction of 24h events first seen in 24h
    "news_burst",             # 24h event count vs trailing 20-session mean
] + [f"news_evt_{t}_w" for t in FEATURED_EVENT_TYPES]  # decayed weight per type

STAGE2_NEWS_FEATURES = [
    "news_articles_2h",
    "news_articles_24h",
    "news_sent_signed_2h",
] + [f"news_evt_{t}_2h" for t in FEATURED_EVENT_TYPES]  # 2h count per type

_SENTIMENT_VALUE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def _epoch_ns(series: pd.Series) -> np.ndarray:
    """Epoch nanoseconds regardless of the series' stored resolution
    (store reads come back as datetime64[s], where a bare astype(int64)
    would silently yield seconds)."""
    return series.astype("datetime64[ns, UTC]").astype("int64").to_numpy()


class NewsFeatureBuilder:
    """Loads and clusters the news store once; serves per-ticker features."""

    def __init__(self, store: NewsStore):
        articles = store.all_articles()
        if not articles.empty:
            clustered = cluster_articles(articles)
            clustered["event_type"] = [
                classify_event(t, d)
                for t, d in zip(clustered["title"], clustered["description"])
            ]
            self._clustered = clustered
        else:
            self._clustered = articles
        self._store = store
        self._per_ticker: dict[str, pd.DataFrame] = {}

    def _ticker_frame(self, ticker: str) -> pd.DataFrame:
        """This ticker's clustered articles sorted by availability."""
        if ticker not in self._per_ticker:
            tagged = self._store.ticker_sentiments(ticker)
            if tagged.empty or self._clustered.empty:
                frame = pd.DataFrame(
                    columns=[
                        "article_id", "published_ts", "available_ts", "publisher",
                        "title", "cluster_id", "is_cluster_start", "sentiment_value",
                        "event_type",
                    ]
                )
            else:
                frame = self._clustered.merge(tagged, on="article_id", how="inner")
                frame["sentiment_value"] = frame["sentiment"].map(_SENTIMENT_VALUE)
                frame = frame.sort_values("available_ts").reset_index(drop=True)
            self._per_ticker[ticker] = frame
        return self._per_ticker[ticker]

    def stage1_features(self, ticker: str, cutoffs: pd.DatetimeIndex) -> pd.DataFrame:
        """One row per cutoff (chronological), indexed like `cutoffs`."""
        frame = self._ticker_frame(ticker)
        # Epoch-nanosecond ints sidestep numpy's tz-aware comparison limits.
        avail = _epoch_ns(frame["available_ts"])
        lam = np.log(2) / DECAY_HALF_LIFE_HOURS
        hour_ns = int(pd.Timedelta(hours=1).value)

        rows = []
        for cutoff in cutoffs:
            cutoff_ns = pd.Timestamp(cutoff).value
            hi = np.searchsorted(avail, cutoff_ns, side="right")
            lo_24 = np.searchsorted(avail, cutoff_ns - 24 * hour_ns, side="right")
            lo_7d = np.searchsorted(avail, cutoff_ns - 24 * 7 * hour_ns, side="right")

            week = frame.iloc[lo_7d:hi]
            day = frame.iloc[lo_24:hi]

            if week.empty:
                rows.append(
                    dict.fromkeys(
                        [f for f in STAGE1_NEWS_FEATURES if f != "news_burst"], 0.0
                    )
                )
                continue

            # Per-cluster view: first visible copy carries the event time;
            # sentiment averages over copies with known sentiment; the
            # cluster's type comes from its earliest copy.
            week_sorted = week.sort_values("available_ts")
            events = week_sorted.groupby("cluster_id").agg(
                first_avail=("available_ts", "min"),
                sent=("sentiment_value", "mean"),
                started_recently=("is_cluster_start", "any"),
                event_type=("event_type", "first"),
            )
            age_hours = (cutoff - events["first_avail"]).dt.total_seconds() / 3600.0
            decay = np.exp(-lam * age_hours.to_numpy())
            sent = events["sent"].fillna(0.0).to_numpy()

            day_clusters = day["cluster_id"].nunique()
            recent = events[events["first_avail"] >= cutoff - pd.Timedelta(hours=24)]
            row = {
                "news_clusters_24h": float(day_clusters),
                "news_clusters_7d": float(len(events)),
                "news_decayed_weight": float(decay.sum()),
                "news_sent_signed": float((decay * sent).sum()),
                "news_sent_magnitude": float((decay * np.abs(sent)).sum()),
                "news_source_diversity": float(day["publisher"].nunique()),
                "news_novelty_frac": float(recent["started_recently"].mean())
                if len(recent)
                else 0.0,
            }
            etypes = events["event_type"].to_numpy()
            for etype in FEATURED_EVENT_TYPES:
                row[f"news_evt_{etype}_w"] = float(decay[etypes == etype].sum())
            rows.append(row)

        out = pd.DataFrame(rows, index=cutoffs)
        # Burst: today's activity vs the stock's own recent normal, strictly
        # backward-looking (shift excludes the current day).
        trailing = out["news_clusters_24h"].shift(1).rolling(20, min_periods=5).mean()
        out["news_burst"] = out["news_clusters_24h"] / (trailing + EPS)
        return out[STAGE1_NEWS_FEATURES]

    def stage2_features(self, ticker: str, bar_ends: pd.DatetimeIndex) -> pd.DataFrame:
        """Vectorized article-window features per bar-end timestamp."""
        frame = self._ticker_frame(ticker)
        avail = _epoch_ns(frame["available_ts"])
        signed = frame["sentiment_value"].fillna(0.0).to_numpy()
        prefix = np.concatenate([[0.0], np.cumsum(signed)])

        if bar_ends.tz is None:
            raise ValueError("bar_ends must be timezone-aware")
        ends = bar_ends.tz_convert("UTC").astype("datetime64[ns, UTC]").asi8
        hour_ns = int(pd.Timedelta(hours=1).value)
        hi = np.searchsorted(avail, ends, side="right")
        lo_2h = np.searchsorted(avail, ends - 2 * hour_ns, side="right")
        lo_24h = np.searchsorted(avail, ends - 24 * hour_ns, side="right")

        columns = {
            "news_articles_2h": (hi - lo_2h).astype(float),
            "news_articles_24h": (hi - lo_24h).astype(float),
            "news_sent_signed_2h": prefix[hi] - prefix[lo_2h],
        }
        etypes = frame["event_type"].to_numpy() if len(frame) else np.array([])
        for etype in FEATURED_EVENT_TYPES:
            type_prefix = np.concatenate(
                [[0.0], np.cumsum((etypes == etype).astype(float))]
            )
            columns[f"news_evt_{etype}_2h"] = type_prefix[hi] - type_prefix[lo_2h]
        return pd.DataFrame(columns, index=bar_ends)
