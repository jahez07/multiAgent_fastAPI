"""
app/agents/claude_client.py - Helper for calling the Anthropic Clade API. 

Wraps the Anthropic SDK with:
    - JSON-structured output via system prompt instructions
    - Timeout and error handling
    - Token usage logging (so costs can be tracked)

Unlike Ollama, Claude doesn't have a native `format: "json"` parameter.
Instead we instruct it in the system prompt to respond ONLY JSON,
and we parse the response.

Usage:
    result = await claude_generate(
        system="You are an analyst...",
        prompt="Analayze this news...",
    )
    # the result is a raw string - caller parses JSON
"""

import json
import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger("claude")

async def claude_generate(
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.2,
        timeout: float = 30.0,
) -> str | None:
    """
    Call Claude API and return the text response.

    Args:
        prompt:         The user message
        system:         System prompt (defines task and output schema)
        max_tokens:     Max response length
        temperature:    Low = more deterministic (good for strucutred output)
        timeout:        Seconds to wait

    Returns:
        The raw text response from Claude or None if anything fails.
        Caller is responsible for JSON parsing.

    Uses httpx directly instead of the anthropic SDK to keep
    dependencies minimal. The API is a simple POST. 
    """
    headers = {
        "x-api-key": settings.anthropic_api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": settings.claude_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
        ],
    }

    if system:
        payload["system"] = system

    try:
        start = time.time()

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()

        elapsed = time.time() - start
        data = resp.json()

        # Extract text from response
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        
        # Log usage for cost tracking
        usage = data.get("usage", {})
        input_tokens = usage.get("input_token", 0)
        output_tokens = usage.get("output_tokens", 0)
        logger.info(
            "Claude: %d in / %d out tokens, %.1fs",
            input_tokens, output_tokens, elapsed,
        )

        if not text.strip():
            logger.warning("    Claude returned empty response")
            return None
        
        return text.strip()
    
    except httpx.TimeoutException:
        logger.error("  Claude request timed out after %.0fs", timeout)
        return None
    
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        body = e.response.text[:300]
        if status == 401:
            logger.error("  Claude API key invalid - check ANTHROPIC_API_KEY in .env")
        elif status == 429:
            logger.warning("    Claude rate limited - consider reducing WORKER_CONCURRENCY")
        elif status == 529:
            logger.warning("    Claude API overloaded = retrying may help")
        else:
            logger.error("  Claude HTTP %d: %s", status, body)
        return None

async def claude_json(
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.2,
) -> dict | None:
    
    """
    Call Claude and parse the response as JSON.

    Handles the common case where Claude wraps JSON in ````json fences.
    Returns parsed dict or None if parsing fails.
    """
    raw = await claude_generate(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if raw is None:
        return None
    
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] # Remove first like (```json)
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0] # Remove last fence
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(
            "   Claude returned invalid JSON: %s - raw: %s",
            e, raw[:200]
        )
        
        return None