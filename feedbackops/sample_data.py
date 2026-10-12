"""
Sample Dataset for FeedbackOps AI.
Contains clearly labeled representative customer feedback reviews for testing and demos.
"""

from typing import List, Dict, Any
import pandas as pd

SAMPLE_REVIEWS: List[Dict[str, Any]] = [
    {
        "id": "REV-001",
        "category": "E-Commerce App",
        "customer": "Sarah Jenkins",
        "source": "App Store Review",
        "date": "2026-10-01",
        "review": (
            "Checkout failed three times during the flash sale! It charged my credit card twice, "
            "but the order status shows as 'Canceled'. Support hasn't responded to my ticket in 24 hours. "
            "Extremely disappointed with this update."
        )
    },
    {
        "id": "REV-002",
        "category": "SaaS Platform",
        "customer": "Alex Mercer",
        "source": "Support Ticket #4892",
        "date": "2026-10-02",
        "review": (
            "The new dashboard layout is sleek and much faster to load. However, the CSV export "
            "feature is missing column headers, which breaks our automated reporting pipeline every morning. "
            "Please fix this header issue ASAP."
        )
    },
    {
        "id": "REV-003",
        "category": "Mobile App",
        "customer": "David Kim",
        "source": "Google Play Review",
        "date": "2026-10-03",
        "review": (
            "Love the app overall! The biometric login feature is super smooth and saves me time daily. "
            "Would love to see dark mode support added in the next release!"
        )
    },
    {
        "id": "REV-004",
        "category": "E-Commerce App",
        "customer": "Elena Rostova",
        "source": "NPS Survey Comment",
        "date": "2026-10-04",
        "review": (
            "Push notifications are spammy and impossible to turn off in settings. The toggle resets "
            "every time I re-open the application. It drains my phone battery and is very intrusive."
        )
    },
    {
        "id": "REV-005",
        "category": "SaaS Platform",
        "customer": "Marcus Thorne",
        "source": "Customer Email",
        "date": "2026-10-05",
        "review": (
            "We were billed the Enterprise tier price despite downgrading to Professional last month. "
            "Our account admin cannot access billing settings due to a permissions error. This needs "
            "immediate escalation to accounting."
        )
    }
]


def get_sample_dataframe() -> pd.DataFrame:
    """Return sample reviews as a pandas DataFrame."""
    return pd.DataFrame(SAMPLE_REVIEWS)


def get_sample_review_by_id(review_id: str) -> Dict[str, Any]:
    """Retrieve sample review by ID."""
    for item in SAMPLE_REVIEWS:
        if item["id"] == review_id:
            return item
    return SAMPLE_REVIEWS[0]
