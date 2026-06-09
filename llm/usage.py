"""
Per-call LLM token-usage logger.

Wraps the OpenAI-shaped `chat.completions.create(...)` surface used by all
pipeline engines and logs input/output token counts after each call so we can
see exactly which stage is the heaviest.

The wrapper is provider-agnostic:
- OpenAI / OpenRouter / Groq all return a response with `usage.prompt_tokens`
  and `usage.completion_tokens`.
- Google's Gemini shim in llm/_provider.py exposes a fake `usage` on its
  `_GeminiResponse` (we patch it there if absent).

Usage:
    from llm.usage import instrumented_create
    response = await instrumented_create(
        stage="planner",
        client=client,
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

Set LLM_LOG_USAGE=0 to silence the per-call log.
"""
import os
import sys
import time
from typing import Any


_LOG_ENABLED = os.getenv("LLM_LOG_USAGE", "1") not in ("0", "false", "False")


def _emit(stage: str, model: str, usage: Any, elapsed_ms: float) -> None:
    if not _LOG_ENABLED:
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if prompt is None and completion is None:
        # Gemini shim or unknown provider — try dict-style
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            total = usage.get("total_tokens")
    print(
        f"[LLM_USAGE] stage={stage:<10} model={model:<45} "
        f"in={prompt} out={completion} total={total} elapsed_ms={elapsed_ms:.1f}",
        file=sys.stderr,
        flush=True,
    )


async def instrumented_create(
    *,
    stage: str,
    client: Any,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    extra_body: dict | None = None,
    **kwargs: Any,
) -> Any:
    """
    Drop-in replacement for `await client.chat.completions.create(...)` that
    logs prompt/completion token counts after the call.

    Pass `extra_body` to forward provider-specific options like
    `{"cache_control": {...}}` for Anthropic, or `prompt_cache_key` for OpenAI.
    """
    create = client.chat.completions.create
    call_kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if extra_body:
        call_kwargs["extra_body"] = extra_body
    call_kwargs.update(kwargs)

    t0 = time.perf_counter()
    response = await create(**call_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    usage = getattr(response, "usage", None)
    _emit(stage, model, usage, elapsed_ms)
    return response
