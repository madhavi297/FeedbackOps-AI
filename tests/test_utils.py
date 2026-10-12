"""
Unit tests for utility functions in FeedbackOps AI.
"""

import json
import pandas as pd
from feedbackops.utils import (
    validate_text_input,
    validate_csv_dataframe,
    compute_aggregate_metrics,
    export_results_to_csv,
    export_results_to_json
)


def test_validate_text_input():
    is_valid, msg = validate_text_input("")
    assert is_valid is False
    assert "cannot be empty" in msg.lower()

    is_valid, msg = validate_text_input("abc")
    assert is_valid is False
    assert "too short" in msg.lower()

    # Oversized review test
    oversized = "a" * 5001
    is_valid, msg = validate_text_input(oversized)
    assert is_valid is False
    assert "exceeds maximum limit" in msg.lower()

    is_valid, msg = validate_text_input("Valid review text about payment issues.")
    assert is_valid is True


def test_validate_csv_dataframe():
    df_empty = pd.DataFrame()
    is_valid, msg, stats = validate_csv_dataframe(df_empty, "review")
    assert is_valid is False
    assert stats["total_rows"] == 0

    df = pd.DataFrame({"feedback": ["Good app", "Bad app", "Good app"]})
    is_valid, msg, stats = validate_csv_dataframe(df, "review")
    assert is_valid is False
    assert "not found" in msg.lower()

    is_valid, msg, stats = validate_csv_dataframe(df, "feedback")
    assert is_valid is True
    assert stats["valid_rows"] == 3
    assert stats["duplicate_rows"] == 1


def test_compute_aggregate_metrics_with_explicit_denominators():
    sample_results = [
        {
            "is_success": True,
            "sentiment": "Negative",
            "overall_urgency": "Critical",
            "urgency_score": 9,
            "pain_points": [{"issue": "Checkout Crash", "is_grounded": True}]
        },
        {
            "is_success": True,
            "sentiment": "Negative",
            "overall_urgency": "High",
            "urgency_score": 7,
            "pain_points": [
                {"issue": "Checkout Crash", "is_grounded": True},
                {"issue": "Slow Loading", "is_grounded": False}
            ]
        },
        {
            "is_success": True,
            "sentiment": "Positive",
            "overall_urgency": "Low",
            "urgency_score": 2,
            "pain_points": []
        },
        {
            # Failed record
            "is_success": False,
            "error": "Timeout occurred"
        }
    ]

    metrics = compute_aggregate_metrics(sample_results)
    assert metrics["total_submitted"] == 4
    assert metrics["successful_count"] == 3
    assert metrics["failed_count"] == 1

    # Denominators are calculated only over successful count (3)
    assert metrics["sentiment_counts"]["Negative"] == 2
    assert metrics["sentiment_percentages"]["Negative"] == 66.7
    assert metrics["sentiment_counts"]["Positive"] == 1
    assert metrics["sentiment_percentages"]["Positive"] == 33.3

    assert metrics["urgency_counts"]["Critical"] == 1
    assert metrics["avg_urgency_score"] == 6.0
    assert metrics["top_pain_points"][0] == {"issue": "Checkout Crash", "count": 2}

    # Grounding rate (2 grounded out of 3 total pain points = 66.7%)
    assert metrics["total_pain_points"] == 3
    assert metrics["grounded_pain_points_pct"] == 66.7


def test_export_results_to_csv_and_json():
    results = [
        {
            "raw_text": "Sample text",
            "sentiment": "Positive",
            "sentiment_score": 0.8,
            "overall_urgency": "Low",
            "urgency_score": 2,
            "summary": "Sample summary",
            "pain_points": [{"urgency": "Low", "issue": "UI design", "evidence_quote": "design"}],
            "action_items": ["Log feature request"],
            "draft_response": "Thank you!",
            "was_sanitized": False,
            "is_success": True
        }
    ]

    csv_str = export_results_to_csv(results)
    assert "Review #" in csv_str
    assert "Sample text" in csv_str
    assert "Positive" in csv_str
    assert "Safety Sanitized" in csv_str

    json_str = export_results_to_json(results)
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert parsed[0]["sentiment"] == "Positive"
