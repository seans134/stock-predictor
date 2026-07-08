"""Duplicate and update clustering.

Syndicated copies of one story must count as one event supported by
multiple sources, not as many independent events. v1 clusters by
normalized title within a 72-hour window: articles whose titles normalize
to the same key and publish within the window share a cluster.

Point-in-time by construction: articles are scanned in publish order and
a cluster id depends only on articles published earlier, so recomputing
with future articles appended never changes an existing assignment.

Deferred (per the plan's initial scope): semantic similarity, shared-fact
matching, and URL canonicalization. Title matching alone catches the
dominant case — verbatim syndication across publishers.
"""

from __future__ import annotations

import re

import pandas as pd

CLUSTER_WINDOW = pd.Timedelta(hours=72)

_WORD = re.compile(r"[a-z0-9]+")


def normalize_title(title: str | None) -> str:
    """Lowercased alphanumeric tokens joined by single spaces."""
    if not title:
        return ""
    return " ".join(_WORD.findall(title.lower()))


def cluster_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Assign cluster_id and is_cluster_start to article rows.

    `articles` needs article_id, published_ts, title. Rows must cover the
    full period being processed; output preserves the input order and adds:
    - cluster_id:       stable id (first article_id in the cluster)
    - is_cluster_start: True for the earliest article of its cluster,
                        i.e. the first time this event was seen (novelty).
    """
    ordered = articles.sort_values("published_ts")
    active: dict[str, tuple[str, pd.Timestamp]] = {}
    cluster_ids: dict[str, str] = {}
    starts: dict[str, bool] = {}

    for row in ordered.itertuples():
        key = normalize_title(row.title)
        published = row.published_ts
        if key:
            existing = active.get(key)
            if existing is not None and published - existing[1] <= CLUSTER_WINDOW:
                cluster_ids[row.article_id] = existing[0]
                starts[row.article_id] = False
                active[key] = (existing[0], published)
                continue
        # Empty titles never cluster; each is its own event.
        cluster_ids[row.article_id] = row.article_id
        starts[row.article_id] = True
        if key:
            active[key] = (row.article_id, published)

    out = articles.copy()
    out["cluster_id"] = out["article_id"].map(cluster_ids)
    out["is_cluster_start"] = out["article_id"].map(starts)
    return out
