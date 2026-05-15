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
from app.databse import log_api_usage, log_error

logger = logging.getLogger("claude")

async def claude_generate(
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.2,
        timeout: float = 30.0,
        model: str | None = None,
        agent_name: str = "unknown",
        pipeline_id: str | None = None,
) -> str | None:
    """
    Call Claude API and return the text response.

    Args:
        prompt:         The user message
        system:         System prompt (defines task and output schema)
        max_tokens:     Max response length
        temperature:    Low = more deterministic (good for strucutred output)
        timeout:        Seconds to wait
        model:          Override model (e.g. setting.claude_haiku for Agent)
        agent_name:     Which agent is calling (for cost tracking)

    Returns:
        The raw text response from Claude or None if anything fails.
        Caller is responsible for JSON parsing.

    Uses httpx directly instead of the anthropic SDK to keep
    dependencies minimal. The API is a simple POST. 
    """

    model = model or settings.claude_model

    headers = {
        "x-api-key": settings.anthropic_api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
        ],
    }

    # Cache the system prompt — it's large and stable across requests.
    # Placing cache_control here (not top-level) ensures the stable prefix
    # is cached rather than the varying user message.
    if system:
        payload["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0

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
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)

        logger.info(
            "Claude: %d in / %d out tokens | cache_read=%d cache_write=%d | %.1fs",
            input_tokens, output_tokens, cache_read, cache_write, elapsed,
        )

        # Log to api_usage table
        await log_api_usage(
            agent=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_time=elapsed,
            success=True,
            pipeline_id=pipeline_id,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

        if not text.strip():
            logger.warning("    Claude returned empty response")
            return None
        
        return text.strip()
    
    except httpx.TimeoutException as e:
        elapsed = time.time() - start
        logger.error("  Claude request timed out after %.0fs", timeout)

        await log_api_usage(
            agent=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=0,
            response_time=elapsed,
            success=False,
            pipeline_id=pipeline_id,
        )

        await log_error(
            "claude_api", e, pipeline_id=pipeline_id,
            context={"agent": agent_name, "model": model, "timeout": timeout},
            severity="error"
        )

        return None
    
    except httpx.HTTPStatusError as e:
        elapsed = time.time() - start
        status = e.response.status_code
        body = e.response.text[:300]

        if status == 401:
            logger.error("  Claude API key invalid - check ANTHROPIC_API_KEY in .env")
            severity = "critical"
        elif status == 429:
            logger.warning("    Claude rate limited - consider reducing WORKER_CONCURRENCY")
            severity = "warning"
        elif status == 529:
            logger.warning("    Claude API overloaded - retrying may help")
            severity = "warning"
        else:
            logger.error("  Claude HTTP %d: %s", status, body)
            severity = "error"
        
        await log_api_usage(
            agent=agent_name, model=model,
            input_tokens=input_tokens, output_tokens=0,
            response_time=elapsed, success=False,
            pipeline_id=pipeline_id,
        )
        await log_error(
            "claude_api", e, pipeline_id=pipeline_id,
            context={"agent": agent_name, "status": status, "body": body},
            severity=severity
        )
        return None
    
    except httpx.ConnectError as e:
        elapsed = time.time() - start
        logger.error("  Cannot connect to Claude API")

        await log_error(
            "claude_api", e, pipeline_id=pipeline_id,
            context={"agent": agent_name},
            severity="critical"
        )

        return None
    
    except Exception as e:
        elapsed = time.time() - start
        logger.error(
            "  Claude unexpected error: %s - %s",
            type(e).__name__, str(e)[:200]
        )

        await log_error(
            "claude_api", e, pipeline_id=pipeline_id,
            context={"agent": agent_name, "model": model},
            severity="error"
        )
        return None

async def claude_json(
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.2,
        timeout: float = 30.0,
        model_name: str | None = None,
        agent_name: str = "unknown",
        pipeline_id: str | None = None,
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
        timeout=timeout,
        model = model_name,
        agent_name=agent_name,
        pipeline_id=pipeline_id,
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

    # Extract just the JSON object - find the outermost { }
    # This handles cases where Claude adds commentary after the JSON
    start = text.find("{")
    if start == -1:
        logger.error("  No JSON object found in response: %s", raw[:200])
        return None
    
    # Find the matching closing brace
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(
            "   Claude returned invalid JSON: %s - raw: %s",
            e, raw[:200]
        )

        await log_error(
            "claude_api", e, pipeline_id=pipeline_id,
            context={"agent": agent_name, "raw_response": raw[:500]},
            severity="warning"
        )
        
        return None