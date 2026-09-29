import sys
from pathlib import Path
import types
from unittest.mock import patch, MagicMock

# Anchor project root to sys.path
KIT_DIR = Path(__file__).resolve().parent.parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

# Ensure dummy modules in sys.modules so patch doesn't fail if packages aren't installed yet
if "groq" not in sys.modules:
    mock_groq_mod = types.ModuleType("groq")
    mock_groq_mod.Groq = MagicMock()
    sys.modules["groq"] = mock_groq_mod

if "google.genai" not in sys.modules:
    if "google" not in sys.modules:
        sys.modules["google"] = types.ModuleType("google")
    mock_genai_mod = types.ModuleType("google.genai")
    mock_genai_mod.Client = MagicMock()
    mock_genai_mod.types = MagicMock()
    sys.modules["google.genai"] = mock_genai_mod

import pytest
from core.llm_client import call_llm, _clean_and_heal_json, LLMResult

def test_json_healing_on_truncated_brace():
    """Verify that a truncated JSON object with missing closing brace is healed cleanly."""
    truncated = '{"name": "hackathon_project", "score": 95'
    ok, parsed, err = _clean_and_heal_json(truncated)
    assert ok is True
    assert parsed["name"] == "hackathon_project"
    assert parsed["score"] == 95

def test_json_healing_with_markdown_fences():
    """Verify markdown fences (```json ... ```) are stripped before parsing."""
    fenced = '```json\n{"status": "ok", "items": [1, 2, 3]}\n```'
    ok, parsed, err = _clean_and_heal_json(fenced)
    assert ok is True
    assert parsed["status"] == "ok"
    assert parsed["items"] == [1, 2, 3]

def test_json_healing_with_unclosed_truncated_fence():
    """Verify markdown fences without closing backticks are handled when truncated."""
    fenced_truncated = '```json\n{"status": "partial", "count": 42'
    ok, parsed, err = _clean_and_heal_json(fenced_truncated)
    assert ok is True
    assert parsed["status"] == "partial"
    assert parsed["count"] == 42

def test_json_healing_with_conversational_preamble():
    """Verify conversational preambles and postscripts are stripped."""
    chatter = 'Here is the JSON payload you requested:\n{"result": "success", "code": 200}\nHope this helps!'
    ok, parsed, err = _clean_and_heal_json(chatter)
    assert ok is True
    assert parsed["result"] == "success"
    assert parsed["code"] == 200

def test_json_unhealable_fails_loud():
    """Verify that fundamentally broken content fails loud rather than guessing field values."""
    broken = "This is not json at all, random string"
    ok, parsed, err = _clean_and_heal_json(broken)
    assert ok is False
    assert parsed is None

@patch("core.llm_client.GROQ_API_KEY", "mock_groq_key")
@patch("core.llm_client.GEMINI_API_KEY", "mock_gemini_key")
@patch("groq.Groq")
def test_fallback_on_groq_429_exhaustion(mock_groq_class):
    """Verify that when Groq raises 429 rate limit across retries, execution cascades to Gemini."""
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat.completions.create.side_effect = Exception("429 Too Many Requests (Rate limit reached)")
    mock_groq_class.return_value = mock_groq_instance

    with patch("google.genai.Client") as mock_gemini_class:
        mock_gemini_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Hello from Gemini fallback!"
        mock_gemini_instance.models.generate_content.return_value = mock_resp
        mock_gemini_class.return_value = mock_gemini_instance

        result = call_llm(system_prompt="Test", user_prompt="Ping")
        assert result.success is True
        assert "Gemini" in result.text
        assert "gemini" in result.provider_used

@patch("core.llm_client.GROQ_API_KEY", "mock_groq_key")
@patch("core.llm_client.GEMINI_API_KEY", "mock_gemini_key")
def test_service_unavailable_when_all_fail():
    """Verify that when all providers fail, SERVICE_UNAVAILABLE is returned gracefully without throwing."""
    with patch("groq.Groq") as mock_groq, patch("google.genai.Client") as mock_gemini:
        mock_groq.side_effect = Exception("Groq network down")
        mock_gemini.side_effect = Exception("Gemini network down")

        result = call_llm(system_prompt="Test", user_prompt="Ping")
        assert result.success is False
        assert result.error == "SERVICE_UNAVAILABLE"
