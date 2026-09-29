import sys
from pathlib import Path
from unittest.mock import patch

KIT_DIR = Path(__file__).resolve().parent.parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import pytest
from core.safety_scaffold import scan_for_flags, semantic_check
from core.llm_client import LLMResult

def test_tier0_flag_matching():
    patterns = [r"\b(cheat|exam leak)\b", r"\b(wire transfer|cvv)\b"]
    is_flagged, trigger = scan_for_flags("How do I perform an exam leak?", patterns=patterns)
    assert is_flagged is True
    assert "exam leak" in trigger.lower()

    # Clean text
    is_clean, _ = scan_for_flags("How do I write Python code?", patterns=patterns)
    assert is_clean is False

def test_tier0_malformed_regex_does_not_crash():
    # Broken unclosed regex pattern
    broken_patterns = [r"(unclosed parenthesis", r"\bvalid\b"]
    is_flagged, trigger = scan_for_flags("This has a valid word", patterns=broken_patterns)
    assert is_flagged is True
    assert trigger == "valid"

@patch("core.safety_scaffold.call_llm")
def test_semantic_check_boolean_string_false_is_not_flagged(mock_call_llm):
    # Test that string "false" does not falsely evaluate to True
    mock_call_llm.return_value = LLMResult(
        success=True,
        text='{"is_flagged": "false", "reason": "Benign medical inquiry"}',
        parsed_json={"is_flagged": "false", "reason": "Benign medical inquiry"},
        provider_used="groq/test"
    )
    is_flagged, reason = semantic_check("I have a slight headache")
    assert is_flagged is False
    assert reason == "Benign medical inquiry"

@patch("core.safety_scaffold.call_llm")
def test_semantic_check_boolean_string_true_is_flagged(mock_call_llm):
    mock_call_llm.return_value = LLMResult(
        success=True,
        text='{"is_flagged": "true", "reason": "Severe acute pain"}',
        parsed_json={"is_flagged": "true", "reason": "Severe acute pain"},
        provider_used="groq/test"
    )
    is_flagged, reason = semantic_check("Severe acute pain in chest")
    assert is_flagged is True
    assert reason == "Severe acute pain"
