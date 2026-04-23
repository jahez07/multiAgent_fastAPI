"""
knowledge_base/embeddings.py - Embedding helper using Ollama. 

Uses the nomic-embed-text model running locally on your GPUs. 
This is called by both ingestion scripts and Agent 3 at query time.

Setup (run once):
    ollama pull nomic-embed-text

Why nomic-embed-text:
    - 768 dimensions (good balance of quality vs storage)
    - Runs fast on GPU (~1ms per embedding)
    - Strong performance on retrieval benchmarks
    - Supports "search_document:" and "search_query:" prefixes
        which improve retrieval accuracy.
"""

import httpx
import logging

from app.config import settings

logger = logging.getLogger("embeddings")


async def embed_text(text: str, prefix: str = "search_document") -> list[float]:
    """
    Embed a single text string using Ollama. 

    Args:
        text: The text to embed
        prefix: "search_document" for ingestion, "search_query" for querying.
                nomic-embed-text uses these prefixes to optimize embeddings
                for their role (document vs query).

    Returns:
        A list of floats (768 dimensions for nomic-embed-text)
    """
    # nomic-embed-text expects a task prefix for best results
    prefixed_text = f"{prefix}: {text}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={
                "model": settings.embedding_model,
                "input": prefixed_text,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns {"embeddings": [[....vector....]]}
        return data["embeddings"][0]
    
async def embed_batch(texts: list[str], prefix: str = "search_document") -> list[list[float]]:
    """
    Embed multiple texts in one call (more efficient than one-by-one).

    Ollama's /api/embed endpoint accepts a list of inputs.
    """
    prefixed = [f"{prefix}:{t}" for t in texts]

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={
                "model": settings.embedding_model,
                "input": prefixed,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        return data["embeddings"]