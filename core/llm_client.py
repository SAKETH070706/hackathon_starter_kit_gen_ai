import re
import json
import time
import base64
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    GROQ_MODELS,
    GROQ_VISION_MODELS,
    GEMINI_MODELS,
    HTTP_TIMEOUT_SECONDS,
    MAX_RETRIES_PER_MODEL,
    logger
)

@dataclass
class LLMResult:
    success: bool
    text: str
    parsed_json: Optional[Dict[str, Any]] = None
    provider_used: str = ""
    error: Optional[str] = None

def _is_transient_error(err_str: str) -> bool:
    """Returns True if error code or message indicates a temporary capacity/rate issue."""
    markers = ("429", "503", "rate", "resource_exhausted", "unavailable", "timeout", "timed out")
    lower_err = err_str.lower()
    return any(m in lower_err for m in markers)

def _heuristic_repair_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts balanced-delimiter repair for truncated JSON outputs.
    Closes unterminated string quotes and unclosed arrays/objects in proper reverse order.
    """
    s = text.strip()
    # Find start of JSON structure
    first_brace = s.find("{")
    first_bracket = s.find("[")
    valid_starts = [i for i in (first_brace, first_bracket) if i != -1]
    if not valid_starts:
        return None
    start_idx = min(valid_starts)
    s = s[start_idx:].rstrip()

    # Track open delimiters while ignoring escaped characters
    in_string = False
    escape = False
    stack = []
    
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    candidate = s
    if in_string:
        candidate += '"'

    # Close unclosed structures in reverse order
    for delimiter in reversed(stack):
        if delimiter == "{":
            candidate += "}"
        elif delimiter == "[":
            candidate += "]"

    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return None

def _clean_and_heal_json(raw_text: str) -> tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Strips markdown code fences (including unclosed fences) and conversational chatter.
    Attempts balanced delimiter repair if the JSON was cut short by max_tokens.
    """
    text = raw_text.strip()
    if not text:
        return False, None, "Empty LLM response"

    # 1. Strip markdown fences (both closed and unclosed/truncated fences)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", text)
    if fence_match and fence_match.group(1).strip():
        text = fence_match.group(1).strip()

    # 2. Pass 1: Standard parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return True, data, ""
        return True, {"data": data}, ""
    except json.JSONDecodeError:
        pass

    # 3. Pass 2: Extract substring between first JSON open and last JSON close
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        snippet = text[first_brace:last_brace + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return True, data, ""
            return True, {"data": data}, ""
        except json.JSONDecodeError:
            pass

    # 4. Pass 3: Balanced delimiter repair for truncated tokens
    repaired_data = _heuristic_repair_json(text)
    if repaired_data is not None:
        return True, repaired_data, ""

    return False, None, f"Failed to parse clean JSON: {raw_text[:200]}"

def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    json_mode: bool = False,
    max_tokens: int = 1000,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/png"
) -> LLMResult:
    """
    Executes an LLM call across Groq with automatic cascade to Gemini.
    Retries only on transient errors (429/503/timeout), with capped backoff.
    """
    # Safeguard for Groq JSON mode requirement: prompt must contain 'json'
    active_sys_prompt = system_prompt
    if json_mode and "json" not in (active_sys_prompt or "").lower() and "json" not in user_prompt.lower():
        active_sys_prompt = (active_sys_prompt + "\nOutput strictly in valid JSON format.") if active_sys_prompt else "Output strictly in valid JSON format."

    # -----------------------------------------------------------------------
    # 1. Tier 1: Groq Cascade
    # -----------------------------------------------------------------------
    if GROQ_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_API_KEY, timeout=HTTP_TIMEOUT_SECONDS)

            # Use vision models if an image is provided; otherwise text models
            candidate_models = GROQ_VISION_MODELS if image_bytes else GROQ_MODELS

            for model_id in candidate_models:
                for attempt in range(MAX_RETRIES_PER_MODEL):
                    try:
                        # Construct content (text or multimodal)
                        if image_bytes:
                            b64 = base64.b64encode(image_bytes).decode("utf-8")
                            user_content = [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                            ]
                        else:
                            user_content = user_prompt

                        messages = []
                        if active_sys_prompt:
                            messages.append({"role": "system", "content": active_sys_prompt})
                        messages.append({"role": "user", "content": user_content})

                        kwargs: Dict[str, Any] = {
                            "model": model_id,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                        if json_mode:
                            kwargs["response_format"] = {"type": "json_object"}

                        resp = groq_client.chat.completions.create(**kwargs)
                        raw_text = (resp.choices[0].message.content or "").strip()

                        if json_mode:
                            ok, parsed, err = _clean_and_heal_json(raw_text)
                            if not ok:
                                return LLMResult(
                                    success=False,
                                    text=raw_text,
                                    provider_used=f"groq/{model_id}",
                                    error=err
                                )
                            logger.info(f"Served by Groq model: {model_id} (JSON)")
                            return LLMResult(
                                success=True,
                                text=raw_text,
                                parsed_json=parsed,
                                provider_used=f"groq/{model_id}"
                            )

                        logger.info(f"Served by Groq model: {model_id}")
                        return LLMResult(
                            success=True,
                            text=raw_text,
                            provider_used=f"groq/{model_id}"
                        )

                    except Exception as err:
                        err_str = str(err)
                        if _is_transient_error(err_str) and attempt < (MAX_RETRIES_PER_MODEL - 1):
                            sleep_time = 0.5 * (attempt + 1)
                            logger.warning(f"Groq {model_id} transient error ({err_str[:80]}), retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                            continue
                        logger.warning(f"Groq {model_id} failed: {err_str[:120]}. Failing to next model.")
                        break
        except Exception as client_err:
            logger.warning(f"Groq client init failed: {client_err}")

    # -----------------------------------------------------------------------
    # 2. Tier 2: Gemini Cascade Fallback
    # -----------------------------------------------------------------------
    if GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types

            g_client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={"timeout": HTTP_TIMEOUT_SECONDS}
            )

            for gem_model in GEMINI_MODELS:
                for attempt in range(MAX_RETRIES_PER_MODEL):
                    try:
                        contents = []
                        if image_bytes:
                            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                        contents.append(user_prompt)

                        config_args: Dict[str, Any] = {
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                        }
                        if active_sys_prompt:
                            config_args["system_instruction"] = active_sys_prompt

                        if json_mode:
                            config_args["response_mime_type"] = "application/json"

                        config = types.GenerateContentConfig(**config_args)
                        g_resp = g_client.models.generate_content(
                            model=gem_model,
                            contents=contents,
                            config=config
                        )
                        raw_text = (g_resp.text or "").strip()

                        if json_mode:
                            ok, parsed, err = _clean_and_heal_json(raw_text)
                            if not ok:
                                return LLMResult(
                                    success=False,
                                    text=raw_text,
                                    provider_used=f"gemini/{gem_model}",
                                    error=err
                                )
                            logger.info(f"Served by Gemini model: {gem_model} (JSON)")
                            return LLMResult(
                                success=True,
                                text=raw_text,
                                parsed_json=parsed,
                                provider_used=f"gemini/{gem_model}"
                            )

                        logger.info(f"Served by Gemini model: {gem_model}")
                        return LLMResult(
                            success=True,
                            text=raw_text,
                            provider_used=f"gemini/{gem_model}"
                        )

                    except Exception as err:
                        err_str = str(err)
                        if _is_transient_error(err_str) and attempt < (MAX_RETRIES_PER_MODEL - 1):
                            sleep_time = 0.5 * (attempt + 1)
                            logger.warning(f"Gemini {gem_model} transient error, retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                            continue
                        logger.warning(f"Gemini {gem_model} failed: {err_str[:120]}. Failing to next model.")
                        break
        except Exception as g_client_err:
            logger.warning(f"Gemini client init failed: {g_client_err}")

    # -----------------------------------------------------------------------
    # 3. Complete Provider Exhaustion
    # -----------------------------------------------------------------------
    logger.critical("All provider endpoints exhausted. Failing soft with SERVICE_UNAVAILABLE.")
    return LLMResult(
        success=False,
        text="",
        error="SERVICE_UNAVAILABLE"
    )
