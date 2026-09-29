# Hackathon Starter Kit Configuration
# Secrets are loaded exclusively from .env - NEVER hardcode API keys or credentials.
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Logging Setup (Centralized)
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hackathon_kit")

# ---------------------------------------------------------------------------
# API Credentials
# ---------------------------------------------------------------------------
# Detect and treat placeholder keys from .env.example as empty
def _clean_api_key(val: str) -> str:
    cleaned = (val or "").strip()
    if not cleaned or "your_" in cleaned.lower() or cleaned.startswith("gsk_your_") or cleaned.startswith("AIzaSy_your_"):
        return ""
    return cleaned

GROQ_API_KEY = _clean_api_key(os.getenv("GROQ_API_KEY", ""))
GEMINI_API_KEY = _clean_api_key(os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""))

# ---------------------------------------------------------------------------
# Provider Model Cascades
# Verified production models across Groq and Google GenAI APIs
# ---------------------------------------------------------------------------
GROQ_MODELS = [
    "llama-3.3-70b-versatile",    # Fast, high-accuracy primary text model
    "llama-3.1-8b-instant",       # Ultra-low latency fallback
    "mixtral-8x7b-32768"          # High-context fallback
]

# Dedicated Groq vision endpoints for multimodal image processing
GROQ_VISION_MODELS = [
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview"
]

GEMINI_MODELS = [
    "gemini-2.5-flash",           # Primary fast multimodal endpoint
    "gemini-2.0-flash",           # Low-latency speed tier
    "gemini-1.5-flash",           # High-throughput stable fallback
    "gemini-1.5-pro"              # Complex reasoning tier
]

# ---------------------------------------------------------------------------
# Resilience & Timeout Policies
# ---------------------------------------------------------------------------
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "25.0"))
MAX_RETRIES_PER_MODEL = int(os.getenv("MAX_RETRIES_PER_MODEL", "2"))

# ---------------------------------------------------------------------------
# Local Storage & Vector Database
# ---------------------------------------------------------------------------
CHROMA_DIR = BASE_DIR / "chroma_store"
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
CHROMA_COLLECTION_NAME = "hackathon_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
