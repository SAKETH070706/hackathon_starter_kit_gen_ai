import re
import json
from typing import List, Tuple
from core.llm_client import call_llm
from config import logger

# ---------------------------------------------------------------------------
# DOMAIN SAFETY PLACEHOLDERS
# FILL IN ONCE PROBLEM STATEMENT IS KNOWN AT THE HACKATHON
# Example for Health: [r"\b(chest pain|heart attack|fainted)\b"]
# Example for Fintech: [r"\b(cvv|otp|wire transfer|stolen card)\b"]
# Example for EdTech: [r"\b(cheat|exam leak|plagiarize)\b"]
# ---------------------------------------------------------------------------
DEFAULT_RED_FLAG_PATTERNS: List[str] = [
    # FILL IN ONCE PROBLEM STATEMENT IS KNOWN
]

DEFAULT_CLASSIFICATION_PROMPT = """You are a real-time safety and triage classifier.
Analyze the user's input for severe policy violations, emergencies, or high-risk requests.
# FILL IN DOMAIN SPECIFIC SAFETY RULES ONCE PROBLEM STATEMENT IS KNOWN

Respond ONLY with a JSON object:
{
  "is_flagged": true or false,
  "reason": "short explanation under 10 words"
}"""

def scan_for_flags(text: str, patterns: List[str] = None) -> Tuple[bool, str]:
    """
    Tier 0 Deterministic Safety Check.
    Ultra-fast (<5ms) regex scan executed before any vector search or LLM call.
    Safeguarded against invalid regex patterns.
    Returns: (is_flagged: bool, matched_phrase: str)
    """
    active_patterns = patterns if patterns is not None else DEFAULT_RED_FLAG_PATTERNS
    if not text or not active_patterns:
        return False, ""

    normalized = text.strip()
    for pattern in active_patterns:
        try:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                trigger = match.group(0)
                logger.warning(f"Tier 0 Safety Flag triggered: '{trigger}'")
                return True, trigger
        except re.error as reg_err:
            logger.error(f"Invalid regex pattern skipped '{pattern}': {reg_err}")
            continue

    return False, ""

def semantic_check(
    text: str,
    classification_prompt: str = DEFAULT_CLASSIFICATION_PROMPT
) -> Tuple[bool, str]:
    """
    Tier 1 Semantic Safety Check.
    Uses LLM JSON mode for contextual classification (e.g. nuanced queries, idioms, vernacular).
    Distinguishes provider failure (SERVICE_UNAVAILABLE) from clean responses.
    Safeguards against string booleans ("true"/"false").
    Returns: (is_flagged: bool, reason: str)
    """
    if not text or len(text.strip()) < 3:
        return False, ""

    res = call_llm(
        system_prompt=classification_prompt,
        user_prompt=f"User input: {text}",
        json_mode=True,
        temperature=0.0,
        max_tokens=150
    )

    if not res.success:
        if res.error == "SERVICE_UNAVAILABLE":
            logger.critical("Tier 1 safety check unreachable: SERVICE_UNAVAILABLE")
            return False, "SERVICE_UNAVAILABLE"
        return False, f"SAFETY_CHECK_ERROR: {res.error}"

    parsed = res.parsed_json or {}
    raw_flag = parsed.get("is_flagged", False)

    # Normalize string boolean values ("true", "false", "1", "0")
    if isinstance(raw_flag, str):
        is_flagged = raw_flag.strip().lower() in ("true", "1", "yes")
    else:
        is_flagged = bool(raw_flag)

    reason = str(parsed.get("reason", ""))
    if is_flagged:
        logger.warning(f"Tier 1 Semantic Safety Flag triggered: {reason}")
    return is_flagged, reason
