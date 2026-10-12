"""
Utility functions for FeedbackOps AI.
Handles input validation, CSV integrity, aggregate metrics computation with explicit denominators,
and data exports.
"""

import json
from typing import Tuple, List, Dict, Any
import pandas as pd


def validate_text_input(text: str, min_len: int = 5, max_len: int = 5000) -> Tuple[bool, str]:
    """
    Validate pasted raw feedback text.
    
    Returns:
        Tuple of (is_valid, error_or_success_message)
    """
    if not text or not isinstance(text, str):
        return False, "Feedback text cannot be empty."

    cleaned = text.strip()
    if len(cleaned) < min_len:
        return False, f"Feedback text is too short. Please provide at least {min_len} characters."

    if len(cleaned) > max_len:
        return False, f"Feedback text exceeds maximum limit of {max_len} characters."

    return True, "Valid text input."


def validate_csv_dataframe(df: pd.DataFrame, target_column: str) -> Tuple[bool, str, Dict[str, int]]:
    """
    Validate uploaded DataFrame for batch feedback analysis.
    Checks for empty files, missing columns, non-null text, and duplicate records.
    
    Returns:
        Tuple of (is_valid, message, stats_dict)
    """
    empty_stats = {"total_rows": 0, "valid_rows": 0, "empty_rows": 0, "duplicate_rows": 0}
    if df is None or df.empty:
        return False, "Uploaded file contains no data rows.", empty_stats

    if target_column not in df.columns:
        cols_str = ", ".join([f"'{c}'" for c in df.columns[:5]])
        return False, f"Column '{target_column}' not found. Available columns: {cols_str}", empty_stats

    total_rows = len(df)
    clean_series = df[target_column].dropna().astype(str).str.strip()
    valid_series = clean_series[clean_series.ne("")]
    valid_count = len(valid_series)
    empty_count = total_rows - valid_count
    duplicate_count = valid_series.duplicated().sum()

    stats = {
        "total_rows": total_rows,
        "valid_rows": valid_count,
        "empty_rows": empty_count,
        "duplicate_rows": int(duplicate_count)
    }

    if valid_count == 0:
        return False, f"Column '{target_column}' has no valid text content.", stats

    msg = f"Found {valid_count} valid feedback entries to process (out of {total_rows} total rows)."
    if duplicate_count > 0:
        msg += f" Note: {duplicate_count} duplicate entries detected."

    return True, msg, stats


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics from batch analysis results.
    Distinguishes successfully processed records from failed ones.
    Computes sentiment distribution, urgency counts, grounding rate, and explicit percentages.
    """
    total_submitted = len(results) if results else 0
    if not results or total_submitted == 0:
        return {
            "total_submitted": 0,
            "successful_count": 0,
            "failed_count": 0,
            "sentiment_counts": {"Positive": 0, "Neutral": 0, "Negative": 0},
            "sentiment_percentages": {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0},
            "urgency_counts": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            "avg_urgency_score": 0.0,
            "grounded_pain_points_pct": 100.0,
            "top_pain_points": []
        }

    successful_results = [r for r in results if r.get("is_success", True) and "error" not in r]
    failed_count = total_submitted - len(successful_results)
    successful_count = len(successful_results)

    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
    urgencies = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    urgency_scores = []
    pain_point_counter: Dict[str, int] = {}
    total_pain_points = 0
    grounded_pain_points = 0

    for r in successful_results:
        sent = r.get("sentiment", "Neutral")
        if sent in sentiments:
            sentiments[sent] += 1
        else:
            sentiments["Neutral"] += 1

        urg = r.get("overall_urgency", "Low")
        if urg in urgencies:
            urgencies[urg] += 1
        else:
            urgencies["Low"] += 1

        score = r.get("urgency_score", 1)
        urgency_scores.append(score)

        for pp in r.get("pain_points", []):
            total_pain_points += 1
            if pp.get("is_grounded", False):
                grounded_pain_points += 1
            issue_title = pp.get("issue", "Unspecified Issue")
            pain_point_counter[issue_title] = pain_point_counter.get(issue_title, 0) + 1

    sorted_pain_points = sorted(pain_point_counter.items(), key=lambda x: x[1], reverse=True)
    avg_score = round(sum(urgency_scores) / successful_count, 1) if successful_count > 0 else 0.0

    # Explicit percentage calculations over successfully processed records
    sentiment_pcts = {}
    for k, v in sentiments.items():
        sentiment_pcts[k] = round((v / successful_count) * 100, 1) if successful_count > 0 else 0.0

    grounding_pct = round((grounded_pain_points / total_pain_points) * 100, 1) if total_pain_points > 0 else 100.0

    return {
        "total_submitted": total_submitted,
        "total_reviews": total_submitted,  # for backward compatibility
        "successful_count": successful_count,
        "failed_count": failed_count,
        "sentiment_counts": sentiments,
        "sentiment_percentages": sentiment_pcts,
        "urgency_counts": urgencies,
        "avg_urgency_score": avg_score,
        "total_pain_points": total_pain_points,
        "grounded_pain_points_pct": grounding_pct,
        "top_pain_points": [{"issue": k, "count": v} for k, v in sorted_pain_points[:10]]
    }


def export_results_to_csv(results: List[Dict[str, Any]]) -> str:
    """Convert batch analysis results to formatted CSV string with evidence and safety status."""
    flattened = []
    for idx, r in enumerate(results, start=1):
        pain_points_summary = "; ".join([
            f"[{p.get('urgency', 'Low')}] {p.get('issue', '')} (Quote: '{p.get('evidence_quote', '')}')" 
            for p in r.get("pain_points", [])
        ])
        actions_summary = " | ".join(r.get("action_items", []))
        
        status = "Success" if r.get("is_success", True) and "error" not in r else f"Failed: {r.get('error', '')}"

        flattened.append({
            "Review #": idx,
            "Status": status,
            "Raw Review": r.get("raw_text", ""),
            "Sentiment": r.get("sentiment", ""),
            "Sentiment Score": r.get("sentiment_score", 0),
            "Overall Urgency": r.get("overall_urgency", ""),
            "Urgency Score (1-10)": r.get("urgency_score", 1),
            "Summary": r.get("summary", ""),
            "Identified Pain Points (with quotes)": pain_points_summary,
            "Recommended Actions (Advisory)": actions_summary,
            "Drafted Customer Response (Draft)": r.get("draft_response", ""),
            "Safety Sanitized": "Yes" if r.get("was_sanitized", False) else "No"
        })

    df = pd.DataFrame(flattened)
    return df.to_csv(index=False)


def export_results_to_json(results: List[Dict[str, Any]]) -> str:
    """Convert batch analysis results to JSON string."""
    return json.dumps(results, indent=2)
