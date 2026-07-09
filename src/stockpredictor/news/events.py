"""Event classification for clustered news.

The plan's initial event categories, implemented as a priority-ordered
keyword classifier over title + description. Deliberately simple and
transparent: every rule is a visible regex, misclassifications are
inspectable, and the classifier is versioned with the feature pipeline
rather than baked into stored data.

Event type is kept separate from sentiment — an earnings event and a
lawsuit can both be "negative" but behave differently. Priority order
resolves multi-topic headlines toward the more market-moving category
(earnings beats analyst chatter, M&A beats product news).
"""

from __future__ import annotations

import re

# Priority order matters: first match wins.
EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "earnings",
        re.compile(
            r"\bearnings\b|\bquarterly (results|report)\b|\bq[1-4]\s+(results|revenue|earnings)"
            r"|\b(beats?|missed?|misses|tops?)\s+(estimates|expectations|wall street)"
            r"|\beps\b|\brevenue (rose|fell|beat|jumped|grew|declined)",
            re.IGNORECASE,
        ),
    ),
    (
        "guidance",
        re.compile(
            r"\bguidance\b|\boutlook\b|\b(raises?|cuts?|lowers?|lifts?|slashes)\s+"
            r"(full[- ]year|annual|quarterly|its)?\s*(forecast|outlook|guidance|view)",
            re.IGNORECASE,
        ),
    ),
    (
        "mna",
        re.compile(
            r"\bmergers?\b|\bacquisitions?\b|\bacquires?\b|\btakeover\b|\bbuyout\b"
            r"|\bto buy\b|\bdeal to acquire\b|\bgoing private\b|\btender offer\b",
            re.IGNORECASE,
        ),
    ),
    (
        "clinical",
        re.compile(
            r"\bfda\b|\bclinical trial\b|\bphase\s+(1|2|3|i{1,3})\b|\bdrug approval\b"
            r"|\bbreakthrough therapy\b|\bnda\b|\bbiologics license\b",
            re.IGNORECASE,
        ),
    ),
    (
        "regulatory_legal",
        re.compile(
            r"\blawsuits?\b|\bsue[sd]?\b|\bsettlements?\b|\bprobes?\b|\binvestigations?\b"
            r"|\bsec charges\b|\bantitrust\b|\bfined\b|\bcourt rul|\bregulators?\b|\bdoj\b",
            re.IGNORECASE,
        ),
    ),
    (
        "management",
        re.compile(
            r"\b(ceo|cfo|coo|chairman|executive|president)\b.*\b(resigns?|steps? down|departs?|"
            r"appoints?|names?|hires?|fires?|ousts?|retires?)\b"
            r"|\b(resigns?|steps? down|appoints?|names?)\b.*\b(ceo|cfo|coo|chairman)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "financing",
        re.compile(
            r"\bshare (offering|repurchase|buyback)\b|\bbuybacks?\b|\bdividends?\b"
            r"|\bsecondary offering\b|\bnotes offering\b|\bconvertible\b|\bdilut"
            r"|\bipo\b|\bstock split\b|\bcapital raise\b",
            re.IGNORECASE,
        ),
    ),
    (
        "analyst",
        re.compile(
            r"\bupgrades?d?\b|\bdowngrades?d?\b|\bprice target\b|\binitiates coverage\b"
            r"|\boverweight\b|\bunderweight\b|\bbuy rating\b|\bsell rating\b|\banalysts?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "product",
        re.compile(
            r"\blaunch(es|ed)?\b|\bunveil(s|ed)?\b|\brecalls?\b|\bnew product\b"
            r"|\bannounces? new\b|\bintroduc(es|ed|ing)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "partnership",
        re.compile(
            r"\bpartnerships?\b|\bcollaborations?\b|\bwins (deal|contract)\b"
            r"|\bcontract (win|award)\b|\bawarded\b|\bsigns? (deal|agreement)\b|\bjoint venture\b",
            re.IGNORECASE,
        ),
    ),
    (
        "macro",
        re.compile(
            r"\bfed\b|\bfederal reserve\b|\binflation\b|\bcpi\b|\bjobs report\b|\bpayrolls\b"
            r"|\btariffs?\b|\binterest rates?\b|\btreasury yields?\b|\brecession\b|\bgdp\b",
            re.IGNORECASE,
        ),
    ),
]

EVENT_TYPES = [name for name, _ in EVENT_PATTERNS] + ["other"]

# The types whose per-type features feed the models; the rest fold into
# the aggregate counts that already exist. Chosen for expected market
# impact, a starting assumption like every other named constant.
FEATURED_EVENT_TYPES = ["earnings", "guidance", "mna", "clinical", "regulatory_legal", "analyst"]


def classify_event(title: str | None, description: str | None = None) -> str:
    """First matching category in priority order; 'other' if none match."""
    text = f"{title or ''} {description or ''}"
    for name, pattern in EVENT_PATTERNS:
        if pattern.search(text):
            return name
    return "other"
