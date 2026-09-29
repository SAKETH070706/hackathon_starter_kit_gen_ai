import os
import glob
import hashlib
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    logger
)

@dataclass
class Chunk:
    text: str
    source_file: str
    distance: float

_CHROMA_COLLECTION = None

def get_chroma_collection(force_reinit: bool = False):
    """Initializes and returns the persistent Chroma collection with local embeddings (cached singleton)."""
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is not None and not force_reinit:
        return _CHROMA_COLLECTION

    if chromadb is None or embedding_functions is None:
        logger.warning("chromadb or sentence-transformers is not installed.")
        return None

    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        _CHROMA_COLLECTION = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
        return _CHROMA_COLLECTION
    except Exception as e:
        logger.warning(f"Could not initialize Chroma collection: {e}")
        return None

def _sliding_window_chunks(text: str, filename: str, chunk_size: int = 600, overlap: int = 90) -> List[dict]:
    """Helper to partition text using a sliding character window."""
    chunks = []
    start = 0
    clean_text = text.strip()
    if not clean_text:
        return []
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        piece = clean_text[start:end].strip()
        if len(piece) > 20:
            chunks.append({"source_file": filename, "text": piece})
        if end >= len(clean_text):
            break
        start += max(1, chunk_size - overlap)
    return chunks

def _chunk_text(content: str, filename: str, chunk_size: int = 600, overlap: int = 90) -> List[dict]:
    """Chunks by markdown headings if present; ensures long sections are subdivided; falls back to sliding window."""
    import re
    # Check for markdown headers (# or ## or ###)
    if re.search(r"(?m)^#{1,3}\s+", content):
        # Split on markdown header lines
        splits = re.split(r"(?m)(?=^#{1,3}\s+)", content)
        chunks = []
        for piece in splits:
            piece = piece.strip()
            if not piece or len(piece) < 20:
                continue
            # If the section is larger than chunk_size, sub-partition it
            if len(piece) > chunk_size:
                chunks.extend(_sliding_window_chunks(piece, filename, chunk_size, overlap))
            else:
                chunks.append({"source_file": filename, "text": piece})
        if chunks:
            return chunks

    return _sliding_window_chunks(content, filename, chunk_size, overlap)

def clear_knowledge() -> bool:
    """Clears all documents from the Chroma collection for a fresh rebuild."""
    global _CHROMA_COLLECTION
    if chromadb is None:
        return False
    try:
        collection = get_chroma_collection()
        if collection is not None:
            # Delete existing collection and recreate
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
            _CHROMA_COLLECTION = None
            get_chroma_collection(force_reinit=True)
            logger.info("Chroma collection cleared successfully.")
            return True
    except Exception as e:
        logger.warning(f"Failed to clear Chroma collection: {e}")
    return False

def ingest_knowledge(folder_path: str, clear_existing: bool = False) -> int:
    """
    Reads all .md and .txt documents from the folder, chunks, and upserts idempotently.
    Returns total chunks ingested. Never crashes on unreadable or empty paths.
    """
    if clear_existing:
        clear_knowledge()

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        logger.warning(f"Knowledge folder does not exist: {folder_path}")
        return 0

    files = glob.glob(str(folder / "*.md")) + glob.glob(str(folder / "*.txt"))
    if not files:
        logger.info(f"No .md or .txt files found in {folder_path}. Ingestion skipped.")
        return 0

    all_chunks = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    all_chunks.extend(_chunk_text(content, Path(fpath).name))
        except Exception as e:
            logger.warning(f"Skipping unreadable file {fpath}: {e}")

    if not all_chunks:
        return 0

    collection = get_chroma_collection()
    if collection is None:
        logger.warning("Chroma collection unavailable; ingestion aborted.")
        return 0

    ids, docs, metas = [], [], []
    for c in all_chunks:
        unique_key = f"{c['source_file']}_{c['text']}"
        chunk_id = hashlib.sha256(unique_key.encode("utf-8")).hexdigest()[:16]
        ids.append(chunk_id)
        docs.append(c["text"])
        metas.append({"source_file": c["source_file"]})

    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    logger.info(f"Ingested {len(ids)} unique chunks into Chroma collection '{CHROMA_COLLECTION_NAME}'.")
    return len(ids)

def retrieve(query: str, k: int = 4, max_distance: float = 0.70) -> List[Chunk]:
    """
    Queries Chroma for relevant chunks.
    NOTE on distance metric: Chroma with hnsw:space='cosine' returns cosine distance.
    Distance = 1.0 - cosine_similarity.
    LOWER = MORE SIMILAR (0.0 = identical, 1.0 = orthogonal).
    Returns an empty list if collection is empty or the best match exceeds max_distance.
    """
    clean_q = query.strip()
    if not clean_q:
        return []

    collection = get_chroma_collection()
    if collection is None:
        return []

    try:
        total_chunks = collection.count()
        if total_chunks == 0:
            logger.info("Chroma collection is empty. Returning zero chunks.")
            return []
    except Exception as e:
        logger.warning(f"Failed to check Chroma collection count: {e}")
        return []

    # Safeguard: Chroma raises ValueError if n_results > collection.count()
    actual_k = min(k, total_chunks)

    try:
        res = collection.query(
            query_texts=[clean_q],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.warning(f"Chroma query failed: {e}")
        return []

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not docs or not dists:
        return []

    # If even the top/closest match exceeds max_distance, refuse to return weak hallucinations
    if dists[0] > max_distance:
        logger.info(f"Best match distance ({dists[0]:.3f}) exceeds threshold ({max_distance}). Suppressing.")
        return []

    results = []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= max_distance:
            results.append(Chunk(
                text=doc,
                source_file=meta.get("source_file", "unknown"),
                distance=round(float(dist), 4)
            ))
    return results
