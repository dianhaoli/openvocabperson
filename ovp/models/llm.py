"""Chat / vision-language model providers.

One small interface for every backend:

* ``openai``     – api.openai.com (GPT-4.1 / GPT-5 family / GPT-4o)
* ``gemini``     – Google Gemini through its OpenAI-compatible endpoint
* ``openrouter`` – any model on OpenRouter
* ``ollama``     – local Ollama server (qwen2.5vl, llava, llama3.2-vision ...)
* ``compat``     – any other OpenAI-compatible server (vLLM, LM Studio, TGI, llama.cpp)
* ``anthropic``  – Claude Messages API
* ``qwen`` / ``hf`` – local Hugging Face transformers VLM (see ``vlm_local.py``)

Messages use an OpenAI-like neutral format; content parts are ``{"type": "text", "text": ...}``
or ``{"type": "image", "image": PIL.Image}``. Providers convert as needed. No vendor SDKs are
required — everything is plain ``httpx`` so the dependency surface stays small.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from PIL import Image

from ..config import Settings

log = logging.getLogger("ovp.llm")


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def encode_image(img: Image.Image, max_side: int = 1024, quality: int = 85) -> str:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply that may contain fences or prose."""
    if text is None:
        raise ValueError("empty reply")
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # first balanced object/array, trying whichever bracket appears first
    pairs = sorted((("{", "}"), ("[", "]")), key=lambda p: (text.find(p[0]) < 0, text.find(p[0])))
    for open_c, close_c in pairs:
        start = text.find(open_c)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except Exception:
                            break
            start = text.find(open_c, start + 1)
    raise ValueError(f"no JSON found in reply: {text[:200]!r}")


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(img: Image.Image) -> dict:
    return {"type": "image", "image": img}


def user(*parts: Any) -> dict:
    content = [text_part(p) if isinstance(p, str) else (image_part(p) if isinstance(p, Image.Image) else p) for p in parts]
    return {"role": "user", "content": content}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


class ProviderError(RuntimeError):
    pass


# ──────────────────────────────────────────────────────────────────────────────
# base
# ──────────────────────────────────────────────────────────────────────────────


class ChatModel:
    provider: str = "base"
    local: bool = False
    supports_tools: bool = True
    supports_vision: bool = True

    def __init__(self, model: str, concurrency: int = 8):
        self.model = model
        self._sem = asyncio.Semaphore(concurrency)

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[ToolSpec]] = None,
        *,
        max_tokens: int = 1024,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> ChatResponse:  # pragma: no cover - abstract
        raise NotImplementedError

    async def complete(self, messages: list[dict], **kw: Any) -> str:
        async with self._sem:
            return (await self.chat(messages, None, **kw)).text

    async def complete_many(self, batch: list[list[dict]], **kw: Any) -> list[str | Exception]:
        """Run many independent requests; cloud providers run them concurrently."""

        async def one(msgs: list[dict]) -> str | Exception:
            try:
                return await self.complete(msgs, **kw)
            except Exception as e:  # keep the batch alive
                log.warning("%s request failed: %s", self.spec, e)
                return e

        return list(await asyncio.gather(*(one(m) for m in batch)))

    async def aclose(self) -> None:
        pass


class _HttpModel(ChatModel):
    def __init__(self, model: str, base_url: str, api_key: Optional[str], timeout: float, concurrency: int):
        super().__init__(model, concurrency)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, url: str, payload: dict, headers: dict) -> dict:
        """POST with retries on 429/5xx and automatic removal of unsupported params."""
        attempt = 0
        payload = dict(payload)
        while True:
            attempt += 1
            try:
                r = await self._client.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt >= 3:
                    raise ProviderError(f"{self.spec}: network error: {e}") from e
                await asyncio.sleep(1.5 * attempt)
                continue
            if r.status_code == 200:
                return r.json()
            body = r.text[:800]
            if r.status_code == 400 and attempt <= 4:
                dropped = self._drop_unsupported(payload, r.text)
                if dropped:
                    log.info("%s: retrying without unsupported param %r", self.spec, dropped)
                    continue
            if r.status_code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < 5:
                retry_after = r.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else (2 ** attempt) * 0.5
                await asyncio.sleep(min(delay, 20) + random.random() * 0.3)
                continue
            raise ProviderError(f"{self.spec}: HTTP {r.status_code}: {body}")

    @staticmethod
    def _drop_unsupported(payload: dict, body: str) -> Optional[str]:
        """Remove (or rename) the request parameter a 400 response complains about.

        Prefers the structured ``error.param`` (OpenAI style); otherwise picks the optional
        parameter mentioned first in the error message. Returns the parameter changed, if any.
        """
        low = body.lower()
        if not any(marker in low for marker in _UNSUPPORTED_MARKERS):
            return None
        mentioned = {key: low.find(alias) for key, aliases in _OPTIONAL_PARAMS.items() for alias in aliases if key in payload and alias in low}
        param = _error_param(body)
        key = param if param in payload and param in _OPTIONAL_PARAMS else min(mentioned, key=mentioned.__getitem__, default=None)
        if key is None:
            return None
        value = payload.pop(key)
        renamed = _TOKEN_PARAM_RENAMES.get(key)
        if renamed and renamed not in payload:
            payload[renamed] = value
        return key


_UNSUPPORTED_MARKERS = ("unsupported", "not supported", "does not support", "unrecognized", "unknown", "extra inputs", "not permitted", "deprecated", "not allowed")
# optional request parameters that may be dropped on a 400, with the spellings error messages use for them
_OPTIONAL_PARAMS: dict[str, tuple[str, ...]] = {
    "reasoning_effort": ("reasoning_effort",),
    "temperature": ("temperature",),
    "response_format": ("response_format",),
    "max_completion_tokens": ("max_completion_tokens",),
    "max_tokens": ("max_tokens",),
    "parallel_tool_calls": ("parallel_tool_calls",),
    "output_config": ("output_config", "effort"),
}
# token-limit parameters are renamed rather than dropped (older servers / reasoning models)
_TOKEN_PARAM_RENAMES = {"max_completion_tokens": "max_tokens", "max_tokens": "max_completion_tokens"}


def _error_param(body: str) -> Optional[str]:
    try:
        err = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return None
    return err.get("param") if isinstance(err, dict) else None


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible (OpenAI, Gemini, OpenRouter, Ollama, vLLM, LM Studio ...)
# ──────────────────────────────────────────────────────────────────────────────


class OpenAICompatModel(_HttpModel):
    def __init__(self, provider: str, model: str, base_url: str, api_key: Optional[str], timeout: float, concurrency: int, image_detail: str = "auto"):
        super().__init__(model, base_url, api_key, timeout, concurrency)
        self.provider = provider
        self.image_detail = image_detail

    @property
    def _is_reasoning(self) -> bool:
        """Models that spend hidden reasoning tokens from the completion budget."""
        m = self.model.lower()
        if self.provider == "openai":
            return m.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))
        if self.provider == "gemini":
            return m.startswith(("gemini-2.5", "gemini-3"))
        return False

    def _convert(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": _as_text(m["content"])})
                continue
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if p["type"] == "text":
                        parts.append({"type": "text", "text": p["text"]})
                    elif p["type"] == "image":
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(p['image'])}", "detail": self.image_detail},
                        })
                # plain text-only content → string (better compatibility with local servers)
                if all(p["type"] == "text" for p in parts):
                    content = "\n".join(p["text"] for p in parts)
                else:
                    content = parts
            msg: dict[str, Any] = {"role": role, "content": content if content is not None else ""}
            if role == "assistant" and m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in m["tool_calls"]
                ]
                if not content:
                    msg["content"] = None
            out.append(msg)
        return out

    async def chat(self, messages, tools=None, *, max_tokens=1024, json_mode=False, temperature=0.0) -> ChatResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": self._convert(messages)}
        budget = max_tokens * REASONING_BUDGET_FACTOR if self._is_reasoning else max_tokens
        payload["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = budget
        if self._is_reasoning:
            payload["reasoning_effort"] = "low"
        if not (self._is_reasoning and self.provider == "openai"):  # OpenAI reasoning models reject temperature
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        data = await self._post(url, payload, headers)
        if self._is_reasoning and _reasoning_exhausted(data):
            # hidden reasoning used the whole budget before any answer: retry once with more room
            token_key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
            payload[token_key] = payload.get(token_key, budget) * 2
            log.info("%s: reasoning exhausted the token budget; retrying with %s=%s", self.spec, token_key, payload[token_key])
            data = await self._post(url, payload, headers)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"{self.spec}: malformed response {str(data)[:300]}") from e
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            calls.append(ToolCall(id=tc.get("id") or f"call_{len(calls)}", name=fn.get("name", ""), arguments=args))
        return ChatResponse(text=(msg.get("content") or "").strip(), tool_calls=calls, raw=data)


REASONING_BUDGET_FACTOR = 4  # completion budget multiplier for models that think before answering


def _reasoning_exhausted(data: dict) -> bool:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return False
    msg = choice.get("message") or {}
    return choice.get("finish_reason") == "length" and not (msg.get("content") or "").strip() and not msg.get("tool_calls")


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic
# ──────────────────────────────────────────────────────────────────────────────


class AnthropicModel(_HttpModel):
    provider = "anthropic"

    def _convert(self, messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_parts.append(_as_text(m["content"]))
                continue
            if role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": _as_text(m["content"])}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) and all(b.get("type") == "tool_result" for b in out[-1]["content"]):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue
            blocks: list[dict] = []
            content = m.get("content")
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for p in content:
                    if p["type"] == "text":
                        blocks.append({"type": "text", "text": p["text"]})
                    elif p["type"] == "image":
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_image(p["image"])}})
            if role == "assistant":
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            if not blocks:
                blocks = [{"type": "text", "text": "(empty)"}]
            out.append({"role": role, "content": blocks})
        return "\n\n".join(system_parts), out

    async def chat(self, messages, tools=None, *, max_tokens=1024, json_mode=False, temperature=0.0) -> ChatResponse:
        system, msgs = self._convert(messages)
        if json_mode:
            system = (system + "\n\nRespond with a single valid JSON value and nothing else.").strip()
        # Current Claude models think by default (adaptive thinking): keep the effort low for these short
        # vision/JSON calls, leave room for thinking in max_tokens, and omit sampling parameters, which
        # they reject when set to a non-default value.
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens * REASONING_BUDGET_FACTOR,
            "messages": msgs,
            "output_config": {"effort": "low"},
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        headers = {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = await self._post(f"{self.base_url}/v1/messages", payload, headers)
        if data.get("stop_reason") == "refusal":
            details = data.get("stop_details") or {}
            raise ProviderError(f"{self.spec}: request refused ({details.get('category') or 'unspecified'})")
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("input") or {}))
        return ChatResponse(text="\n".join(texts).strip(), tool_calls=calls, raw=data)


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return json.dumps(content, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# factory
# ──────────────────────────────────────────────────────────────────────────────

CLOUD_PROVIDERS = ("openai", "anthropic", "gemini", "openrouter", "ollama", "compat")
LOCAL_PROVIDERS = ("qwen", "hf")


def split_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        return spec, ""
    provider, model = spec.split(":", 1)
    return provider.strip().lower(), model.strip()


def provider_available(provider: str, s: Settings) -> tuple[bool, Optional[str]]:
    if provider == "openai":
        return (bool(s.openai_api_key), None if s.openai_api_key else "set OPENAI_API_KEY")
    if provider == "anthropic":
        return (bool(s.anthropic_api_key), None if s.anthropic_api_key else "set ANTHROPIC_API_KEY")
    if provider == "gemini":
        return (bool(s.gemini_api_key), None if s.gemini_api_key else "set GEMINI_API_KEY")
    if provider == "openrouter":
        return (bool(s.openrouter_api_key), None if s.openrouter_api_key else "set OPENROUTER_API_KEY")
    if provider == "compat":
        return (bool(s.compat_base_url), None if s.compat_base_url else "set OVP_COMPAT_BASE_URL")
    if provider == "ollama":
        return (True, "requires a running Ollama server")
    if provider in LOCAL_PROVIDERS:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return (False, "pip install torch transformers accelerate")
        return (True, None)
    return (False, f"unknown provider {provider!r}")


def create_chat_model(spec: str, s: Settings) -> ChatModel:
    provider, model = split_spec(spec)
    t, c = s.request_timeout, s.vlm_concurrency
    if provider == "openai":
        return OpenAICompatModel("openai", model or s.openai_vlm_model, s.openai_base_url, s.openai_api_key, t, c)
    if provider == "gemini":
        return OpenAICompatModel("gemini", model or s.gemini_model, s.gemini_base_url, s.gemini_api_key, t, c)
    if provider == "openrouter":
        return OpenAICompatModel("openrouter", model, "https://openrouter.ai/api/v1", s.openrouter_api_key, t, c)
    if provider == "ollama":
        return OpenAICompatModel("ollama", model or "qwen2.5vl:7b", s.ollama_base_url, "ollama", max(t, 300), 2)
    if provider == "compat":
        if not s.compat_base_url:
            raise ProviderError("OVP_COMPAT_BASE_URL is not set")
        return OpenAICompatModel("compat", model, s.compat_base_url, s.compat_api_key, t, c)
    if provider == "anthropic":
        return AnthropicModel(model or s.anthropic_vlm_model, s.anthropic_base_url, s.anthropic_api_key, t, c)
    if provider in LOCAL_PROVIDERS:
        from .vlm_local import LocalVLM

        return LocalVLM(model or s.local_vlm_model, s)
    raise ProviderError(f"unknown model provider {provider!r} (spec {spec!r})")
