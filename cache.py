"""Core caching decorator and SemantiCache class."""

import asyncio
import functools
import hashlib
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np

from .embedder import get_default_embedder
from .stats import _global_stats
from .store import BaseStore, MemoryStore, SQLiteStore

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="semanticache_embed_")


def _default_key_fn(*args, **kwargs) -> str:
    if args:
        return str(args[0])
    if "prompt" in kwargs:
        return str(kwargs["prompt"])
    if kwargs:
        return str(next(iter(kwargs.values())))
    raise ValueError("No arguments provided to extract prompt from.")


class SemantiCache:
    def __init__(
        self,
        threshold: float = 0.92,
        backend: Literal["memory", "sqlite"] = "memory",
        path: str = "./cache.db",
        ttl: Optional[int] = None,
        max_size: Optional[int] = None,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
        key_fn: Optional[Callable[..., str]] = None,
        cost_per_call: Optional[float] = None,
    ):
        self.threshold = threshold
        self.ttl = ttl
        self.max_size = max_size
        self.embedder = embedder or get_default_embedder()
        self.key_fn = key_fn or _default_key_fn
        self.cost_per_call = cost_per_call

        if backend == "sqlite":
            self.store: BaseStore = SQLiteStore(path=path)
        else:
            self.store: BaseStore = MemoryStore()

    def _evict_if_needed(self):
        if self.ttl is not None:
            self.store.evict_expired(self.ttl)
        if self.max_size is not None and self.store.size() > self.max_size:
            self.store.evict_oldest(self.store.size() - self.max_size)

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder(text)

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get_or_call(self, prompt: str, fn: Callable[[], Any]) -> Any:
        self._evict_if_needed()

        # 1. Exact-match fast path (no embedding needed)
        ph = self._prompt_hash(prompt)
        exact = self.store.get_by_hash(ph)
        if exact is not None:
            latency_ms = (exact.metadata or {}).get("_latency_ms", 0)
            _global_stats.record_hit(
                cost_per_call=self.cost_per_call, latency_ms=latency_ms
            )
            return exact.response

        # 2. Embed and semantic search
        embedding = self._embed(prompt)
        similar = self.store.find_similar(embedding, threshold=self.threshold, ttl=self.ttl)
        if similar is not None:
            latency_ms = (similar.metadata or {}).get("_latency_ms", 0)
            _global_stats.record_hit(
                cost_per_call=self.cost_per_call, latency_ms=latency_ms
            )
            return similar.response

        # 3. Miss — call function and cache
        t0 = time.perf_counter()
        result = fn()
        latency_ms = (time.perf_counter() - t0) * 1000
        _global_stats.record_miss()

        self.store.add(
            prompt,
            embedding,
            str(result),
            metadata={"_latency_ms": latency_ms},
        )
        self._evict_if_needed()
        return result

    async def aget_or_call(
        self, prompt: str, fn: Callable, fn_args: tuple = (), fn_kwargs: Optional[Dict] = None
    ) -> Any:
        self._evict_if_needed()

        # 1. Exact-match fast path (no embedding needed)
        ph = self._prompt_hash(prompt)
        exact = self.store.get_by_hash(ph)
        if exact is not None:
            latency_ms = (exact.metadata or {}).get("_latency_ms", 0)
            _global_stats.record_hit(
                cost_per_call=self.cost_per_call, latency_ms=latency_ms
            )
            return exact.response

        # 2. Embed and semantic search (run embedding in thread pool)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(_executor, self._embed, prompt)

        similar = self.store.find_similar(embedding, threshold=self.threshold, ttl=self.ttl)
        if similar is not None:
            latency_ms = (similar.metadata or {}).get("_latency_ms", 0)
            _global_stats.record_hit(
                cost_per_call=self.cost_per_call, latency_ms=latency_ms
            )
            return similar.response

        # 3. Miss
        t0 = time.perf_counter()
        if inspect.iscoroutinefunction(fn):
            result = await fn(*fn_args, **(fn_kwargs or {}))
        else:
            result = fn(*fn_args, **(fn_kwargs or {}))
        latency_ms = (time.perf_counter() - t0) * 1000
        _global_stats.record_miss()

        self.store.add(
            prompt,
            embedding,
            str(result),
            metadata={"_latency_ms": latency_ms},
        )
        self._evict_if_needed()
        return result


def cache(
    threshold: float = 0.92,
    backend: Literal["memory", "sqlite"] = "memory",
    path: str = "./cache.db",
    ttl: Optional[int] = None,
    max_size: Optional[int] = None,
    embedder: Optional[Callable[[str], np.ndarray]] = None,
    key_fn: Optional[Callable[..., str]] = None,
    cost_per_call: Optional[float] = None,
):
    def decorator(fn: Callable) -> Callable:
        sc = SemantiCache(
            threshold=threshold,
            backend=backend,
            path=path,
            ttl=ttl,
            max_size=max_size,
            embedder=embedder,
            key_fn=key_fn or _default_key_fn,
            cost_per_call=cost_per_call,
        )

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                prompt = sc.key_fn(*args, **kwargs)
                return await sc.aget_or_call(prompt, fn, args, kwargs)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                prompt = sc.key_fn(*args, **kwargs)
                return sc.get_or_call(prompt, lambda: fn(*args, **kwargs))
            return sync_wrapper

    return decorator
