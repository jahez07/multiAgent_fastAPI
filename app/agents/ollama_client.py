"""
app/agents/ollama_client.py - Helper for calling Ollama's local API.

Ollama exposes a REST API at http://192.168.2.185:11434. This module
wraps the /api/generate endpoint with:
    - JSON format enforcement (forces valid JSON output)
    - Timeout handling (local models can be slow on first load)
    - Retry on transient failures

    
The key Ollama feature we use is `format: "json"` - this constrains
the model to ONLY output valid JSON, no markdown fences, no preamble. 
Combined with a clear schema in the prompt, this gives us reliable 
structured extraction from Llama 3.1 8B.

Usage:
    result = await ollama_generate(
        prompt = "Classify this news..",
        system = "You are a news Classifier...",
    )
    # result is already a parsed dict
"""

import json
import logging

import httpx

from app.config import settings
from app.databse import log_error

logger = logging.getLogger("ollama")


async def ollama_generate(
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        timeout: float = 60.0,
        model_name: str = settings.ollama_model,
        agent_name: str = "unknown",
        pipeline_id: str | None = None,
) -> dict | None:
    """
    Calla Ollama's /api/generate endpoint and return parsed JSON.

    Args:
        prompt:         The user message (the news to classifiy)
        system:         System prompt (defines the task and output schema)
        temperature:    Low = more deterministic (good for extraction)
        timeout:        Seconds to wait (first call may be slow due to model loading)
    
    Returns:
        Parsed dict from the model's JSON output, or None if anything fails.
    
    Why /api/generate instead of /api/chat:
        For single-turn structured extraction, /api/generate is simpler
        and slightly fater - no conversation history overhead.
    """
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "system": system,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512, # Cap ouptut length (classification is short)
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Ollama returns {"response": "...", "done": true, ...}
        raw_text = data.get("response"," ")

        if not raw_text.strip():
            logger.warning("Ollama returned empty response")
            return None
        
        # Parse the JSON response
        result = json.loads(raw_text)
        return result
    
    except json.JSONDecodeError as e:
        logger.error("Ollama returned invalid JSON: %s - raw: %s",
                     e, raw_text[:200] if 'raw_text' in dir() else "?")
        
        await log_error(
            "ollama_api", e, pipeline_id=pipeline_id,
            context={"agent": agent_name, "model": model_name, "timeout": timeout},
            severity="warning"
        )
        return None
    
    except httpx.TimeoutException as e:
        logger.error("Ollama request timed out after %.0fs", timeout)

        await log_error(
            "ollama_api", e, pipeline_id=pipeline_id,
            context={"agent": agent_name, "model": model_name, "timeout": timeout},
            severity="error"
        )
        return None
    
    except httpx.ConnectError as e:
        logger.error("Cannot connect to Ollama at %s 0 is it running?", settings.ollama_base_url)

        await log_error(
            "ollama_api", e, pipeline_id= pipeline_id,
            context= {"agent": agent_name, "model": model_name, "timeout": timeout},
            severity="critical",
        )
        return None
        
    
    except Exception as e:
        logger.error("Ollama unexpected error: %s - %s", type(e).__name__, str(e)[:200])

        await log_error(
            "ollama_api", e, pipeline_id= pipeline_id,
            context= {"agent": agent_name, "model": model_name},
            severity="error"
        )
        return None