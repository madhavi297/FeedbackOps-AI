"""
Integration test for live Ollama inference.
Automatically detects whether local Ollama is reachable and verifies end-to-end inference and safety.
"""

import pytest
import requests
from feedbackops.ollama_client import OllamaClient
from feedbackops.analyzer import FeedbackAnalyzer


def is_live_ollama_ready() -> bool:
    """Check if local Ollama daemon is reachable and has llama3.2."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any("llama3.2" in m for m in models)
        return False
    except Exception:
        return False


LIVE_OLLAMA_AVAILABLE = is_live_ollama_ready()


@pytest.mark.skipif(not LIVE_OLLAMA_AVAILABLE, reason="Local Ollama daemon with llama3.2 is not running")
def test_live_ollama_real_inference():
    """Verify real Ollama inference and schema output with llama3.2."""
    client = OllamaClient(host="http://localhost:11434", timeout=120)
    analyzer = FeedbackAnalyzer(client=client, model="llama3.2")

    test_review = (
        "I was billed twice for my subscription last week. The app crashed at checkout "
        "and I want my money refunded immediately!"
    )

    result = analyzer.analyze_review(test_review, warmup=True)

    assert result["sentiment"] in ["Negative", "Neutral"]
    assert -1.0 <= result["sentiment_score"] <= 1.0
    assert result["overall_urgency"] in ["High", "Critical", "Medium"]
    assert 1 <= result["urgency_score"] <= 10
    assert len(result["summary"]) > 0

    # Verify that the draft response does NOT contain unauthorized commitments
    assert "within 24 hours" not in result["draft_response"].lower()
    assert "we have refunded" not in result["draft_response"].lower()
    assert "refund has been issued" not in result["draft_response"].lower()

    # Verify evidence grounding structure
    for pp in result.get("pain_points", []):
        assert "issue" in pp
        assert "evidence_quote" in pp
        assert isinstance(pp["is_grounded"], bool)
