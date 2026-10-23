"""FastAPI demo server for semanticache."""

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..cache import SemantiCache
from ..stats import get_stats, _global_stats

app = FastAPI(title="semanticache demo")

# Global demo cache (SQLite so it persists across restarts)
demo_cache = SemantiCache(
    threshold=0.92,
    backend="sqlite",
    path="./semanticache_demo.db",
    ttl=None,
    max_size=1000,
    cost_per_call=0.02,
)


def mock_llm(prompt: str) -> str:
    """Simulate an LLM call with ~200ms latency and a templated response."""
    time.sleep(0.2)
    return (
        f"This is a mock LLM response to your prompt.\n\n"
        f"You asked: \"{prompt}\".\n"
        f"Since this is a demo, no real API call was made. "
        f"The semanticache layer intercepted and cached this result."
    )


class AskRequest(BaseModel):
    prompt: str


class AskResponse(BaseModel):
    response: str
    cache_hit: bool
    similarity_score: Optional[float] = None
    latency_ms: float


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    prompt = req.prompt.strip()

    # 1. Exact-match fast path
    ph = _prompt_hash(prompt)
    exact = demo_cache.store.get_by_hash(ph)
    cache_hit = False
    similarity_score = None
    response_text = None
    latency_ms = 200.0

    if exact is not None:
        cache_hit = True
        similarity_score = 1.0
        response_text = exact.response
        latency_ms = (exact.metadata or {}).get("_latency_ms", 200)

    if not cache_hit:
        # 2. Semantic search
        embedding = demo_cache._embed(prompt)
        similar = demo_cache.store.find_similar(
            embedding, threshold=demo_cache.threshold, ttl=demo_cache.ttl
        )
        if similar is not None:
            cache_hit = True
            similarity_score = float(np.dot(embedding, similar.embedding))
            response_text = similar.response
            latency_ms = (similar.metadata or {}).get("_latency_ms", 200)

    if cache_hit:
        _global_stats.record_hit(
            cost_per_call=demo_cache.cost_per_call, latency_ms=latency_ms
        )
    else:
        t1 = time.perf_counter()
        response_text = mock_llm(prompt)
        latency_ms = (time.perf_counter() - t1) * 1000
        _global_stats.record_miss()
        embedding = demo_cache._embed(prompt)
        demo_cache.store.add(
            prompt, embedding, response_text, metadata={"_latency_ms": latency_ms}
        )
        demo_cache._evict_if_needed()

    total_latency = (time.perf_counter() - t0) * 1000
    return {
        "response": response_text,
        "cache_hit": cache_hit,
        "similarity_score": round(similarity_score, 4) if similarity_score is not None else None,
        "latency_ms": round(total_latency, 2),
    }


@app.get("/stats")
def stats_endpoint() -> Dict[str, Any]:
    return get_stats()


@app.get("/entries")
def entries_endpoint(limit: int = 20) -> List[Dict[str, Any]]:
    return demo_cache.store.entries(limit=limit)


# Serve static frontend files
import os
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
