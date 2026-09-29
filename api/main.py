import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Ensure starter kit root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from core.llm_client import call_llm
from core.rag import retrieve, ingest_knowledge, clear_knowledge
from core.extraction import extract_structured_data
from core.safety_scaffold import scan_for_flags, semantic_check
from config import KNOWLEDGE_DIR

app = FastAPI(
    title="GenAI Hackathon Starter API",
    version="1.0.0",
    description="Domain-agnostic resilient API with dual-provider LLM cascade & Chroma RAG."
)

# Standard CORS configuration (allow_credentials=False when using wildcard "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query or task")
    user_context: Optional[str] = ""
    run_semantic_guardrail: Optional[bool] = False

class QueryResponse(BaseModel):
    status: str
    response: str
    provider_used: str
    sources: List[str] = Field(default_factory=list)

class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int
    message: str

class DefaultExtractSchema(BaseModel):
    entity_name: str
    category: str
    key_points: List[str]
    confidence_score: float

class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=3)

class ExtractResponse(BaseModel):
    status: str
    extracted: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "hackathon-starter-api"}

@app.post("/api/ingest", response_model=IngestResponse)
def trigger_knowledge_ingest(clear_first: bool = False):
    """Triggers knowledge base document ingestion from data/knowledge folder."""
    count = ingest_knowledge(str(KNOWLEDGE_DIR), clear_existing=clear_first)
    return IngestResponse(
        status="success",
        chunks_ingested=count,
        message=f"Successfully ingested {count} chunks into Chroma vector store."
    )

@app.post("/api/process", response_model=QueryResponse)
def process_user_query(req: QueryRequest):
    """Orchestrates Tier 0/1 safety scan, Chroma vector retrieval, and LLM synthesis."""
    # 1. Tier 0 Safety Scan (Deterministic)
    is_flagged, trigger = scan_for_flags(req.query)
    if is_flagged:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security/Policy violation triggered (Tier 0): '{trigger}'"
        )

    # Optional: Tier 1 Semantic Safety Scan
    if req.run_semantic_guardrail:
        is_semantic_flagged, reason = semantic_check(req.query)
        if is_semantic_flagged:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Security/Policy violation triggered (Tier 1): '{reason}'"
            )

    # 2. Vector Retrieval
    chunks = retrieve(req.query, k=3, max_distance=0.70)
    context_str = "\n\n".join([f"[{c.source_file}]:\n{c.text}" for c in chunks]) if chunks else "No relevant context found."
    source_files = list(dict.fromkeys([c.source_file for c in chunks]))

    # 3. LLM Call
    system_prompt = (
        "You are an intelligent hackathon solution assistant. "
        "Ground your advice strictly on verified context.\n\n"
        f"VERIFIED CONTEXT:\n{context_str}"
    )
    result = call_llm(system_prompt=system_prompt, user_prompt=req.query)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service temporarily unavailable: {result.error}"
        )

    return QueryResponse(
        status="ok",
        response=result.text,
        provider_used=result.provider_used,
        sources=source_files
    )

@app.post("/api/extract", response_model=ExtractResponse)
def extract_fields_endpoint(req: ExtractRequest):
    """Generic structured extraction endpoint with feedback retry."""
    res = extract_structured_data(schema=DefaultExtractSchema, text=req.text)
    if res.success and res.validated:
        return ExtractResponse(status="success", extracted=res.validated.model_dump())
    return ExtractResponse(status="failed", error=res.error)

@app.post("/api/extract/image", response_model=ExtractResponse)
async def extract_image_endpoint(file: UploadFile = File(...)):
    """Multimodal extraction from uploaded image using schema validation with retry."""
    file_bytes = await file.read()
    mime_type = file.content_type or "image/png"
    res = extract_structured_data(
        schema=DefaultExtractSchema,
        image_bytes=file_bytes,
        mime_type=mime_type
    )
    if res.success and res.validated:
        return ExtractResponse(status="success", extracted=res.validated.model_dump())
    return ExtractResponse(status="failed", error=res.error)
