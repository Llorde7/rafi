"""
Shared async chat-completion client factory used by the pipeline engines
(classifier, causal, planner, trace).

The provider is selected via the `LLM_PROVIDER` env var:
  - "openrouter" (default) -> AsyncOpenAI pointed at https://openrouter.ai/api/v1
                              model: OPENROUTER_MODEL (default: meta-llama/llama-3.3-70b-instruct:free)
  - "groq"                 -> AsyncGroq (existing behaviour)
  - "openai"               -> AsyncOpenAI pointed at api.openai.com
  - "gemini"               -> google-genai (sync, wrapped)
  - "anthropic"            -> not implemented

All engines call `await client.chat.completions.create(...)` with the same
OpenAI-shaped surface, so swapping providers is a one-env-var change.
"""
import os
from typing import Any

from dotenv import load_dotenv

# Ensure local .env overrides process env when present
load_dotenv(override=True)


def get_provider() -> str:
    # Default to 'groq' for safety if no provider is set in production envs
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def get_default_model() -> str:
    provider = get_provider()
    if provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Lazy async clients
# ---------------------------------------------------------------------------

_async_client: Any | None = None


def _build_async_client() -> Any:
    """Build (and cache) the right async client for the current provider."""
    global _async_client
    if _async_client is not None:
        return _async_client

    provider = get_provider()

    if provider == "openrouter":
        from openai import AsyncOpenAI
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("LLM provider 'openrouter' selected but OPENROUTER_API_KEY is not set")
        _async_client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    elif provider == "groq":
        from groq import AsyncGroq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("LLM provider 'groq' selected but GROQ_API_KEY is not set")
        _async_client = AsyncGroq(api_key=key)
    elif provider == "openai":
        from openai import AsyncOpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("LLM provider 'openai' selected but OPENAI_API_KEY is not set")
        _async_client = AsyncOpenAI(api_key=key)
    elif provider == "gemini":
        # Gemini's SDK is sync; we expose a thin async shim with the same
        # chat.completions.create(...) surface so engine code is unchanged.
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("LLM provider 'gemini' selected but GEMINI_API_KEY is not set")
        _async_client = _GeminiAsyncShim(genai.Client(api_key=key))
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return _async_client


def get_async_chat_client() -> Any:
    """Return the async client. Engines use this in place of AsyncGroq(...)."""
    return _build_async_client()


def reset_client_cache() -> None:
    """Useful for tests when env vars change at runtime."""
    global _async_client
    _async_client = None


# ---------------------------------------------------------------------------
# Gemini shim — exposes chat.completions.create(messages=..., ...) on top of
# google-genai's sync generate_content, so engines can stay provider-agnostic.
# ---------------------------------------------------------------------------

class _GeminiAsyncShim:
    def __init__(self, genai_client):
        import asyncio
        self._client = genai_client
        self._asyncio = asyncio

    class _Completions:
        def __init__(self, outer):
            self._outer = outer

        class _Create:
            def __init__(self, outer):
                self._outer = outer

            async def create(self, *, model, messages, temperature, max_tokens, stream=False):
                system_prompt = ""
                contents = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system_prompt = msg.get("content", "")
                    elif msg.get("role") == "user":
                        contents.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
                    elif msg.get("role") == "assistant":
                        contents.append({"role": "model", "parts": [{"text": msg.get("content", "")}]})

                config = {"temperature": temperature, "max_output_tokens": max_tokens}
                if system_prompt:
                    config["system_instruction"] = system_prompt

                if stream:
                    return self._stream(model, contents, config)

                response = await self._outer._asyncio.to_thread(
                    self._outer._client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                )
                return _GeminiResponse(response)

            async def _stream(self, model, contents, config):
                # Gemini streaming in this SDK version: emit a single final chunk.
                response = await self._outer._asyncio.to_thread(
                    self._outer._client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                )
                yield _GeminiChunk(text=response.text or "")

        def create(self, *, model, messages, temperature, max_tokens, stream=False):
            return self._Create(self._outer).create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

    @property
    def chat(self):
        class _Chat:
            def __init__(self, outer):
                self.completions = outer._Completions(outer)
        return _Chat(self)


class _GeminiResponse:
    def __init__(self, response):
        self.choices = [_GeminiChoice(response)]


class _GeminiChoice:
    def __init__(self, response):
        self.message = _GeminiMessage(response)


class _GeminiMessage:
    def __init__(self, response):
        self.content = (response.text or "").strip()


class _GeminiChunk:
    def __init__(self, text):
        self.choices = [_GeminiDeltaChoice(text)]


class _GeminiDeltaChoice:
    def __init__(self, text):
        class _Delta:
            def __init__(self, t):
                self.content = t
        self.delta = _Delta(text)
