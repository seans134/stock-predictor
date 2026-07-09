import pytest

from stockpredictor.news.events import EVENT_TYPES, FEATURED_EVENT_TYPES, classify_event


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Acme Q3 earnings beat estimates as revenue rose 12%", "earnings"),
        ("Acme raises full-year guidance after strong quarter", "guidance"),
        ("MegaCorp to buy Acme in $5 billion takeover", "mna"),
        ("Acme wins FDA approval for Phase 3 drug candidate", "clinical"),
        ("SEC charges Acme executives in accounting probe", "regulatory_legal"),
        ("Acme CEO steps down; board names interim chief", "management"),
        ("Acme announces $2 billion share buyback and dividend hike", "financing"),
        ("Morgan Stanley upgrades Acme, lifts price target to $95", "analyst"),
        ("Acme unveils next-generation widget lineup", "product"),
        ("Acme signs agreement with BigCo for joint venture", "partnership"),
        ("Fed holds interest rates steady as inflation cools", "macro"),
        ("Ten stocks I like this week", "other"),
    ],
)
def test_classify_event_categories(title, expected):
    assert classify_event(title) == expected


def test_priority_earnings_beats_analyst():
    # Multi-topic headline resolves to the more market-moving category.
    title = "Analysts react as Acme earnings beat expectations"
    assert classify_event(title) == "earnings"


def test_description_is_searched_too():
    assert classify_event("Acme update", "Company raised its full-year outlook") == "guidance"
    assert classify_event(None, None) == "other"


def test_event_type_lists_consistent():
    assert set(FEATURED_EVENT_TYPES) <= set(EVENT_TYPES)
    assert "other" in EVENT_TYPES
