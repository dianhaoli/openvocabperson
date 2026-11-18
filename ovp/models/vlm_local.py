"""Local Hugging Face VLM (Qwen2.5-VL / Qwen3-VL / SmolVLM / Gemma 3 / ... via transformers).

Improvements over the original pipeline:
* generic ``AutoModelForImageTextToText`` loading – any chat-template VLM works
* no ``output_hidden_states`` during generation (it held every layer for every step)
* true batched generation with left padding, size-aware batch packing
* runs on the device worker thread – never blocks the event loop
* lazily loaded on first use, optional 8/4-bit quantization (CUDA)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from PIL import Image

from ..config import Settings
from ..runtime import get_runtime
from .llm import ChatModel, ChatResponse, ProviderError

log = logging.getLogger("ovp.vlm_local")

MAX_PIXELS = 512 * 28 * 28  # caps visual tokens per image (Qwen convention)


def batch_by_area(items: list[tuple[int, int]], max_batch: int, max_area: int) -> list[list[int]]:
    """First-fit-decreasing packing of (index, area) so one huge crop can't stall a batch."""
    order = sorted(items, key=lambda x: x[1], reverse=True)
    batches: list[tuple[list[int], int]] = []
    for idx, area in order:
        for b in batches:
            if len(b[0]) < max_batch and b[1] + area <= max_area:
                b[0].append(idx)
                batches[batches.index(b)] = (b[0], b[1] + area)
                break
        else:
            batches.append(([idx], area))
    return [b[0] for b in batches]


class LocalVLM(ChatModel):
    provider = "qwen"
    local = True
    supports_tools = False

    def __init__(self, model: str, settings: Settings):
        super().__init__(model, concurrency=1)
        self.settings = settings
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    # loading happens on the device thread
    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as AutoVLM
            except ImportError:  # older transformers
                from transformers import AutoModelForVision2Seq as AutoVLM

            device = get_runtime().device
            kwargs: dict[str, Any] = {"torch_dtype": torch.float16 if device != "cpu" else torch.float32}
            if device.startswith("cuda"):
                kwargs["device_map"] = "auto"
                try:
                    import flash_attn  # noqa: F401

                    kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    kwargs["attn_implementation"] = "sdpa"
                q = self.settings.vlm_quantization
                if q in ("8bit", "4bit"):
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = (
                        BitsAndBytesConfig(load_in_8bit=True)
                        if q == "8bit"
                        else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
                    )
                if torch.cuda.is_bf16_supported():
                    kwargs["torch_dtype"] = torch.bfloat16
            log.info("loading local VLM %s on %s", self.model, device)
            model = AutoVLM.from_pretrained(self.model, **kwargs)
            if not device.startswith("cuda"):
                model = model.to(device)
            model.eval()
            processor = AutoProcessor.from_pretrained(self.model, max_pixels=MAX_PIXELS)
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                tok.padding_side = "left"
                if tok.pad_token_id is None:
                    tok.pad_token_id = tok.eos_token_id
            self._model, self._processor = model, processor

    async def warmup(self) -> None:
        await get_runtime().run_model(self._load)

    @staticmethod
    def _hf_messages(messages: list[dict]) -> tuple[list[dict], list[Image.Image]]:
        out, images = [], []
        for m in messages:
            if m["role"] == "tool":
                continue
            content = m.get("content")
            if isinstance(content, str):
                out.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
                continue
            parts = []
            for p in content or []:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image":
                    parts.append({"type": "image"})
                    images.append(p["image"].convert("RGB"))
            out.append({"role": m["role"], "content": parts})
        return out, images

    def _generate_batch(self, batch: list[list[dict]], max_tokens: int) -> list[str]:
        import torch

        self._load()
        proc, model = self._processor, self._model
        texts, images = [], []
        for msgs in batch:
            hf, imgs = self._hf_messages(msgs)
            texts.append(proc.apply_chat_template(hf, tokenize=False, add_generation_prompt=True))
            images.extend(imgs)
        inputs = proc(text=texts, images=images or None, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        tok = getattr(proc, "tokenizer", proc)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1] :]
        return [s.strip() for s in proc.batch_decode(gen, skip_special_tokens=True)]

    async def chat(self, messages, tools=None, *, max_tokens=256, json_mode=False, temperature=0.0) -> ChatResponse:
        if tools:
            raise ProviderError("local VLM does not support tool calling; use an API or Ollama model for the agent")
        texts = await get_runtime().run_model(self._generate_batch, [messages], max_tokens)
        return ChatResponse(text=texts[0])

    async def complete_many(self, batch: list[list[dict]], **kw: Any) -> list[str | Exception]:
        max_tokens = kw.get("max_tokens", 256)
        if not batch:
            return []

        def area(msgs: list[dict]) -> int:
            total = 0
            for m in msgs:
                if isinstance(m.get("content"), list):
                    for p in m["content"]:
                        if p["type"] == "image":
                            w, h = p["image"].size
                            total += min(w * h, MAX_PIXELS)
            return max(total, 1)

        groups = batch_by_area([(i, area(m)) for i, m in enumerate(batch)], self.settings.vlm_batch_size, self.settings.vlm_batch_size * MAX_PIXELS)
        results: list[Optional[str | Exception]] = [None] * len(batch)
        for g in groups:
            try:
                outs = await get_runtime().run_model(self._generate_batch, [batch[i] for i in g], max_tokens)
                for i, o in zip(g, outs):
                    results[i] = o
            except Exception as e:  # fall back to one-by-one for this group
                log.warning("batched generation failed (%s); retrying individually", e)
                for i in g:
                    try:
                        results[i] = (await get_runtime().run_model(self._generate_batch, [batch[i]], max_tokens))[0]
                    except Exception as e2:
                        results[i] = e2
        return results  # type: ignore[return-value]
