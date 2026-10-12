"""
Unit tests for OllamaClient API integration, timeout handling, and model warm-up.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import requests

from feedbackops.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaModelNotFoundError,
    OllamaResponseError
)


def test_client_configurable_timeout():
    client_default = OllamaClient()
    assert client_default.timeout == 180

    client_custom = OllamaClient(host="http://localhost:11434", timeout=300)
    assert client_custom.timeout == 300


def test_check_connection_success():
    client = OllamaClient()
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        is_conn, msg = client.check_connection()
        assert is_conn is True
        assert "running" in msg.lower()


def test_check_connection_failure():
    client = OllamaClient()
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("Refused")

        is_conn, msg = client.check_connection()
        assert is_conn is False
        assert "could not connect" in msg.lower()


def test_list_models_success():
    client = OllamaClient()
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:latest"}
            ]
        }
        mock_get.return_value = mock_response

        models = client.list_models()
        assert "llama3.2:latest" in models
        assert "mistral:latest" in models


def test_is_model_available():
    client = OllamaClient()
    with patch.object(client, "list_models", return_value=["llama3.2:latest", "phi3"]):
        assert client.is_model_available("llama3.2") is True
        assert client.is_model_available("llama3.2:latest") is True
        assert client.is_model_available("phi3") is True
        assert client.is_model_available("gpt-4") is False


def test_generate_json_success():
    client = OllamaClient(timeout=180)
    expected_dict = {
        "sentiment": "Negative",
        "sentiment_score": -0.8,
        "summary": "Checkout failed",
        "pain_points": [],
        "overall_urgency": "High",
        "urgency_score": 8,
        "action_items": ["Fix payment gateway"],
        "draft_response": "We apologize for the error."
    }

    with patch.object(client, "check_connection", return_value=(True, "OK")), \
         patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps(expected_dict)
        }
        mock_post.return_value = mock_response

        result = client.generate_json(model="llama3.2", prompt="Analyze review")
        assert result["sentiment"] == "Negative"
        assert result["urgency_score"] == 8
        assert result["action_items"] == ["Fix payment gateway"]

        # Verify exact URL and timeout argument passed to requests.post
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == (10, 180)


def test_generate_json_timeout_handling():
    """Verify that requests.exceptions.Timeout raises OllamaTimeoutError with timeout info."""
    client = OllamaClient(timeout=180)

    with patch.object(client, "check_connection", return_value=(True, "OK")), \
         patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out. (read timeout=180)")

        with pytest.raises(OllamaTimeoutError) as exc_info:
            client.generate_json(model="llama3.2", prompt="Analyze review")

        assert "180 seconds" in str(exc_info.value)
        assert "timed out" in str(exc_info.value).lower()


def test_generate_json_codeblock_parsing():
    client = OllamaClient()
    wrapped_json = "```json\n{\"sentiment\": \"Positive\", \"urgency_score\": 2}\n```"

    with patch.object(client, "check_connection", return_value=(True, "OK")), \
         patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": wrapped_json}
        mock_post.return_value = mock_response

        result = client.generate_json(model="llama3.2", prompt="Test")
        assert result["sentiment"] == "Positive"
        assert result["urgency_score"] == 2


def test_generate_json_model_not_found():
    client = OllamaClient()
    with patch.object(client, "check_connection", return_value=(True, "OK")), \
         patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        with pytest.raises(OllamaModelNotFoundError):
            client.generate_json(model="unknown_model", prompt="Test")


def test_warmup_model():
    client = OllamaClient()
    with patch.object(client, "check_connection", return_value=(True, "OK")), \
         patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        is_ready, msg = client.warmup_model("llama3.2")
        assert is_ready is True
        assert "warm and ready" in msg
