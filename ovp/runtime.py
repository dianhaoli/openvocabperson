"""Compute runtime: device selection and non-blocking execution of model code.

Design:
* All *local* model inference for a device runs on one dedicated worker thread. PyTorch
  CUDA/MPS work is serialized there (no thread-safety surprises, no GPU over-subscription)
  and the asyncio event loop is never blocked — the server stays responsive while models run.
* CPU-bound helpers (image decode/encode, numpy) go to a small shared thread pool.
* Cloud calls are plain asyncio (httpx) and run concurrently, bounded by semaphores.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

log = logging.getLogger("ovp")
T = TypeVar("T")


def detect_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch missing → cloud-only mode
        pass
    return "cpu"


def configure_torch(device: str) -> None:
    try:
        import torch
    except Exception:
        return
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device == "cpu":
        # leave a core for the event loop / IO
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


class Runtime:
    def __init__(self, device: str):
        self.device = device
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ovp-{device}")
        self._cpu = ThreadPoolExecutor(
            max_workers=min(8, (os.cpu_count() or 2) + 2), thread_name_prefix="ovp-cpu"
        )

    async def run_model(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run local-model code on the device worker thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._gpu, functools.partial(fn, *args, **kwargs))

    async def run_cpu(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._cpu, functools.partial(fn, *args, **kwargs))

    def shutdown(self) -> None:
        self._gpu.shutdown(wait=False, cancel_futures=True)
        self._cpu.shutdown(wait=False, cancel_futures=True)


_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        from .config import get_settings

        device = detect_device(get_settings().device)
        configure_torch(device)
        _runtime = Runtime(device)
    return _runtime
