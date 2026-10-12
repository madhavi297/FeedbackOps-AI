"""
Safety and Compliance Enforcement Module for FeedbackOps AI.
Prevents unauthorized refund promises, unapproved commitments, invented deadlines,
and ensures evidence-based grounding of AI outputs.
"""

import re
from typing import Tuple, List, Optional


# Regex patterns that indicate unauthorized business commitments, deadlines, or completed actions
UNAUTHORIZED_PATTERNS = [
    # Refund and payout promises
    r"\b(?:we\s+have|we've)\s+(?:issued|processed|sent|initiated|approved)\s+(?:a|your)?\s*refund\b",
    r"\b(?:will|shall)\s+(?:issue|process|send|refund|return)\s+(?:your\s+money|the\s+money|a\s+full\s+refund|your\s+refund|the\s+charge)\b",
    r"\b(?:refund|chargeback)\s+(?:has\s+been|is\s+now)\s+(?:approved|processed|issued)\b",
    r"\b(?:you\s+will\s+receive|expect)\s+(?:a\s+refund|your\s+money\s+back)\b",
    r"\bcredited\s+back\s+to\s+your\s+(?:account|card|bank)\b",
    
    # Invented deadlines
    r"\bwithin\s+\d+\s*(?:hours|hrs|minutes|days|business\s+days)\b",
    r"\bby\s+(?:tomorrow|tonight|the\s+end\s+of\s+today|this\s+evening)\b",
    r"\bguarantee(?:d)?\s+(?:within|by|resolution)\b",
    
    # Unauthorized compensation / vouchers
    r"\b(?:provide|offer|give|send|credited?|added?)\s+(?:you\s+)?(?:with\s+)?(?:a\s+)?(?:free\s+)?(?:voucher|coupon|gift\s*card|credit|discount|compensation)\b",
    r"\b(?:free\s+voucher|free\s+coupon|free\s+credit|monetary\s+compensation)\b",
    r"\bcompensat(?:e|ion)\s+(?:you\s+)?for\s+(?:the|your)\s+(?:trouble|inconvenience|delay|issue)\b",
    
    # Invented claims of already-completed actions
    r"\b(?:we\s+have|we've)\s+(?:fixed|resolved|corrected|repaired)\s+the\s+(?:bug|issue|glitch|problem)\b",
    r"\b(?:replacement|order)\s+has\s+already\s+been\s+shipped\b"
]

COMPILED_UNAUTHORIZED_REGEX = [re.compile(p, re.IGNORECASE) for p in UNAUTHORIZED_PATTERNS]


# Safe default template when model hallucinates unauthorized commitments
SAFE_ACKNOWLEDGMENT_TEMPLATE = (
    "Thank you for reaching out and sharing your feedback. We sincerely apologize for the frustration "
    "and inconvenience you experienced. We have documented your concern and forwarded it to our customer "
    "support and operations team for review. A team member will investigate your account details "
    "and follow up with you as soon as possible."
)


def detect_unauthorized_commitments(text: str) -> Tuple[bool, List[str]]:
    """
    Check if a text string contains unauthorized refund promises, compensation, or deadlines.
    
    Args:
        text: Text string to inspect (e.g. model-generated draft response)
        
    Returns:
        Tuple of (has_unauthorized_commitment: bool, detected_snippets: List[str])
    """
    if not text or not isinstance(text, str):
        return False, []

    detected = []
    for regex in COMPILED_UNAUTHORIZED_REGEX:
        matches = regex.findall(text)
        if matches:
            detected.extend(matches)

    return len(detected) > 0, detected


def sanitize_draft_response(draft_text: str, review_text: str = "") -> Tuple[str, bool, List[str]]:
    """
    Sanitize and enforce safety on customer response drafts.
    If unauthorized commitments (e.g. unapproved refund, deadlines) are found,
    the response is safely replaced with a compliant acknowledgment.
    
    Args:
        draft_text: Raw draft response from the model
        review_text: Original customer review text
        
    Returns:
        Tuple of (safe_response: str, was_sanitized: bool, detected_violations: List[str])
    """
    if not draft_text or not isinstance(draft_text, str) or not draft_text.strip():
        return SAFE_ACKNOWLEDGMENT_TEMPLATE, True, ["empty_draft"]

    has_violation, violations = detect_unauthorized_commitments(draft_text)
    if has_violation:
        # Construct safe replacement that acknowledges the issue without unauthorized commitments
        safe_text = SAFE_ACKNOWLEDGMENT_TEMPLATE
        return safe_text, True, violations

    return draft_text.strip(), False, []


def verify_evidence_excerpt(excerpt: str, source_text: str) -> Tuple[bool, str]:
    """
    Verify whether an excerpt or quote extracted by the model actually exists in the source text.
    Performs case-insensitive and normalized whitespace verification.
    
    Args:
        excerpt: Substring excerpt extracted by model
        source_text: Original raw customer review
        
    Returns:
        Tuple of (is_grounded: bool, clean_excerpt: str)
    """
    if not excerpt or not isinstance(excerpt, str) or not source_text or not isinstance(source_text, str):
        return False, ""

    clean_exc = excerpt.strip().strip('"\'')
    if not clean_exc:
        return False, ""

    # Direct substring check
    if clean_exc in source_text:
        return True, clean_exc

    # Case-insensitive substring check
    lower_source = source_text.lower()
    lower_exc = clean_exc.lower()
    if lower_exc in lower_source:
        idx = lower_source.find(lower_exc)
        matched_str = source_text[idx:idx + len(clean_exc)]
        return True, matched_str

    # Normalized whitespace check
    norm_source = " ".join(source_text.split()).lower()
    norm_exc = " ".join(clean_exc.split()).lower()
    if norm_exc in norm_source:
        return True, clean_exc

    return False, clean_exc


def sanitize_action_item(action: str) -> str:
    """
    Ensure action items are clearly labeled as recommendations rather than completed actions.
    
    Args:
        action: Raw action item string
        
    Returns:
        Properly labeled recommendation string
    """
    if not action or not isinstance(action, str):
        return ""

    cleaned = action.strip()
    
    # Remove phrases suggesting the action was already completed
    cleaned = re.sub(r"^(?:have|already|completed:?)\s*", "", cleaned, flags=re.IGNORECASE)
    
    # If action is phrased as a refund command, frame it as policy review
    if re.search(r"\b(?:refund|money\s+back)\b", cleaned, re.IGNORECASE):
        if not re.search(r"\b(?:review|investigate|assess|check|verify)\b", cleaned, re.IGNORECASE):
            cleaned = f"Review account logs to assess refund eligibility under policy: {cleaned}"

    return cleaned


def detect_prompt_injection(text: str) -> bool:
    """
    Detect potential adversarial prompt injection attempts in customer review input.
    
    Args:
        text: Raw customer review input
        
    Returns:
        True if text contains suspicious system override patterns
    """
    if not text or not isinstance(text, str):
        return False

    suspicious_patterns = [
        r"ignore\s+(?:all\s+)?previous\s+instructions",
        r"disregard\s+(?:all\s+)?prior\s+prompts",
        r"system\s*:\s*you\s+are",
        r"you\s+are\s+now\s+an\s+unrestricted",
        r"you\s+must\s+approve\s+a\s+refund",
        r"output\s+only\s+the\s+following"
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
