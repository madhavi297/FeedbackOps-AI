"""
Ollama API Client for FeedbackOps AI.
Handles connections, health checks, model verification, warm-up, and structured JSON inference.
"""

import json
import re
from typing import Any, Dict, List, Tuple, Optional
import requests


class OllamaError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when Ollama daemon is unreachable."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama generation request times out."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when specified model is not pulled in Ollama."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when response generation or JSON parsing fails."""
    pass


class OllamaClient:
    """Client for interacting with local Ollama REST API."""

    def __init__(self, host: str = "http://localhost:11434", timeout: int = 180):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama server base URL (default: http://localhost:11434)
            timeout: Request read timeout in seconds (default: 180)
        """
        self.host = host.rstrip("/")
        self.timeout = timeout

    def check_connection(self) -> Tuple[bool, str]:
        """
        Check if local Ollama service is running and reachable.
        
        Returns:
            Tuple of (is_connected, message)
        """
        try:
            url = f"{self.host}/api/tags"
            response = requests.get(url, timeout=(5, 10))
            if response.status_code == 200:
                return True, "Ollama server is running and reachable."
            return False, f"Ollama returned HTTP status {response.status_code}."
        except requests.exceptions.Timeout:
            return False, f"Connection check to Ollama at {self.host} timed out."
        except requests.exceptions.ConnectionError:
            return False, (
                f"Could not connect to Ollama at {self.host}. "
                "Ensure Ollama application or service is running."
            )
        except Exception as e:
            return False, f"Unexpected error connecting to Ollama: {str(e)}"

    def list_models(self) -> List[str]:
        """
        Fetch list of locally installed model names.
        
        Returns:
            List of model names (e.g. ['llama3.2:latest', 'mistral'])
        
        Raises:
            OllamaConnectionError: If server cannot be reached.
            OllamaTimeoutError: If request times out.
        """
        try:
            url = f"{self.host}/api/tags"
            response = requests.get(url, timeout=(5, 15))
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return models
            raise OllamaConnectionError(f"HTTP {response.status_code} when fetching models.")
        except requests.exceptions.Timeout:
            raise OllamaTimeoutError("Fetching model list from Ollama timed out.")
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Failed to fetch model list: {str(e)}")

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if specific model is available in Ollama.
        Matches exact name or prefix (e.g. 'llama3.2' matches 'llama3.2:latest').
        """
        try:
            models = self.list_models()
            clean_target = model_name.split(":")[0].lower()
            for m in models:
                m_clean = m.split(":")[0].lower()
                if m.lower() == model_name.lower() or m_clean == clean_target:
                    return True
            return False
        except OllamaError:
            return False

    def warmup_model(self, model: str, warmup_timeout: int = 15) -> Tuple[bool, str]:
        """
        Perform a lightweight model health check / warm-up before full analysis.
        Checks model availability and triggers quick loading into memory without blocking indefinitely.
        
        Args:
            model: Model name to warm up
            warmup_timeout: Timeout limit for warm-up check in seconds
            
        Returns:
            Tuple of (is_ready, message)
        """
        is_conn, conn_msg = self.check_connection()
        if not is_conn:
            return False, conn_msg

        # Step 1: Verify model metadata via /api/show
        try:
            show_url = f"{self.host}/api/show"
            show_resp = requests.post(show_url, json={"name": model}, timeout=(5, warmup_timeout))
            if show_resp.status_code == 404:
                return False, f"Model '{model}' is not pulled in Ollama."
        except requests.exceptions.RequestException:
            pass  # Fall through to quick ping test

        # Step 2: Send 1-token generate request to warm up VRAM/RAM
        try:
            gen_url = f"{self.host}/api/generate"
            payload = {
                "model": model,
                "prompt": "ping",
                "stream": False,
                "options": {"num_predict": 1}
            }
            resp = requests.post(gen_url, json=payload, timeout=(5, warmup_timeout))
            if resp.status_code == 200:
                return True, f"Model '{model}' is warm and ready."
            elif resp.status_code == 404:
                return False, f"Model '{model}' not found."
            return False, f"Warmup returned HTTP status {resp.status_code}."
        except requests.exceptions.Timeout:
            return False, f"Model warm-up timed out after {warmup_timeout} seconds, but continuing with main request."
        except requests.exceptions.RequestException as e:
            return False, f"Model warm-up failed: {str(e)}"

    def generate_json(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from Ollama.
        
        Args:
            model: Name of local model (e.g., 'llama3.2')
            prompt: User prompt text
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Parsed JSON object as Python dictionary
            
        Raises:
            OllamaConnectionError: If server unreachable
            OllamaTimeoutError: If generation request times out
            OllamaModelNotFoundError: If model not pulled
            OllamaResponseError: If generation fails or response invalid JSON
        """
        is_conn, conn_msg = self.check_connection()
        if not is_conn:
            raise OllamaConnectionError(conn_msg)

        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            # Connect timeout = 10s, Read timeout = self.timeout (default 180s)
            response = requests.post(url, json=payload, timeout=(10, self.timeout))
        except requests.exceptions.Timeout as te:
            raise OllamaTimeoutError(
                f"Ollama generation timed out after {self.timeout} seconds. "
                "The local model took too long to respond. Try increasing the timeout setting or choosing a smaller model."
            )
        except requests.exceptions.ConnectionError as ce:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.host}. Ensure Ollama application or service is running."
            )
        except requests.exceptions.RequestException as re:
            raise OllamaConnectionError(f"Network error during Ollama generation: {str(re)}")

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{model}' not found in local Ollama instance. "
                f"Run 'ollama pull {model}' in terminal to download it."
            )
        elif response.status_code != 200:
            raise OllamaResponseError(
                f"Ollama returned HTTP error code {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            raw_text = data.get("response", "").strip()
        except Exception as e:
            raise OllamaResponseError(f"Failed to parse API response wrapper: {str(e)}")

        if not raw_text:
            raise OllamaResponseError("Ollama returned an empty response.")

        # Parse structured JSON output
        parsed = self._parse_json_string(raw_text)
        if parsed is not None:
            return parsed

        raise OllamaResponseError(
            f"Failed to parse structured JSON from Ollama output: {raw_text[:200]}..."
        )

    def _parse_json_string(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON content from model output."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code blocks if Ollama wrapped output in ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Regex search for first { ... } block
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None
