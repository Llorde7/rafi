import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMClient(ABC):
    """Abstract LLM client interface for clean provider switching."""

    @abstractmethod
    def chat_completion_sync(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Sync non-streaming chat completion. Returns response content string."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Async non-streaming chat completion. Returns response content string."""
        pass

    @abstractmethod
    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        """Streaming chat completion. Yields token chunks."""
        pass


class GroqClient(LLMClient):
    def __init__(self):
        from groq import Groq, AsyncGroq
        self._sync_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._async_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    def chat_completion_sync(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self._sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        stream = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"token": chunk.choices[0].delta.content, "done": False}
        yield {"token": "", "done": True}


class OpenAIClient(LLMClient):
    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat_completion_sync(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError("Use async chat_completion for OpenAI")

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"token": chunk.choices[0].delta.content, "done": False}
        yield {"token": "", "done": True}


class GeminiClient(LLMClient):
    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def chat_completion_sync(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Sync implementation for Gemini
        system_prompt = ""
        contents = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                contents.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
            elif msg.get("role") == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.get("content", "")}]})

        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_prompt:
            config["system_instruction"] = system_prompt

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return response.text.strip() if response.text else ""

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Gemini SDK is sync only - run in thread
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chat_completion_sync(model, messages, temperature, max_tokens)
        )

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        # Gemini streaming
        import asyncio
        system_prompt = ""
        contents = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                contents.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
            elif msg.get("role") == "model":
                contents.append({"role": "model", "parts": [{"text": msg.get("content", "")}]})

        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_prompt:
            config["system_instruction"] = system_prompt

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )
        full_response = response.text or ""
        # Yield as single chunk (Gemini doesn't support true streaming in this SDK version)
        yield {"token": full_response, "done": True}


def get_llm_client(provider: str = None) -> tuple[LLMClient, str]:
    """
    Get configured LLM client and default model.
    
    Priority: LLM_PROVIDER env var > default (groq)
    Returns: (client, default_model)
    """
    provider = provider or os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == LLMProvider.GROQ:
        return GroqClient(), os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif provider == LLMProvider.OPENAI:
        return OpenAIClient(), os.getenv("OPENAI_MODEL", "gpt-4o")
    elif provider == LLMProvider.GEMINI:
        return GeminiClient(), os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    elif provider == LLMProvider.ANTHROPIC:
        raise NotImplementedError("Anthropic client not yet implemented")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")