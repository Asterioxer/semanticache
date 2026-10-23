"""Comprehensive tests for semanticache."""

import asyncio
import hashlib
import os
import tempfile
import time

import numpy as np
import pytest

from semanticache import cache, SemantiCache, get_stats
from semanticache.stats import reset_stats
from semanticache.store import MemoryStore, SQLiteStore


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fake_embedder(text: str) -> np.ndarray:
    """Deterministic fake embedder for fast, reproducible tests."""
    np.random.seed(hash(text) & 0xFFFFFFFF)
    vec = np.random.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def _near_match_embedder(text: str) -> np.ndarray:
    """Fake embedder where 'hello' prompts cluster and 'world' prompts cluster."""
    base = np.zeros(384, dtype=np.float32)
    t = text.lower()
    if "hello" in t:
        base[0] = 1.0
    elif "world" in t:
        base[1] = 1.0
    else:
        base[2] = 1.0
    np.random.seed(hash(text) & 0xFFFFFFFF)
    noise = np.random.randn(384).astype(np.float32) * 0.005
    vec = base + noise
    vec = vec / np.linalg.norm(vec)
    return vec


# ─── Exact Match Tests ──────────────────────────────────────────────────────

def test_exact_match_hit():
    reset_stats()
    call_count = [0]

    @cache(threshold=0.92, embedder=_fake_embedder)
    def ask(prompt: str) -> str:
        call_count[0] += 1
        return f"answer to {prompt}"

    r1 = ask("what is the capital of france")
    r2 = ask("what is the capital of france")
    assert r1 == r2 == "answer to what is the capital of france"
    assert call_count[0] == 1
    stats = get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1


def test_exact_match_miss():
    reset_stats()
    call_count = [0]

    @cache(threshold=0.92, embedder=_fake_embedder)
    def ask(prompt: str) -> str:
        call_count[0] += 1
        return f"answer to {prompt}"

    r1 = ask("question one")
    r2 = ask("question two")
    assert call_count[0] == 2
    assert r1 != r2


# ─── Semantic Near-Match Tests ────────────────────────────────────────────────

def test_semantic_near_match_hit_above_threshold():
    reset_stats()
    call_count = [0]

    @cache(threshold=0.92, embedder=_near_match_embedder)
    def ask(prompt: str) -> str:
        call_count[0] += 1
        return "cached response"

    r1 = ask("hello there")
    r2 = ask("hello everyone")
    assert call_count[0] == 1
    assert r1 == r2 == "cached response"


def test_semantic_miss_below_threshold():
    reset_stats()
    call_count = [0]

    @cache(threshold=0.92, embedder=_near_match_embedder)
    def ask(prompt: str) -> str:
        call_count[0] += 1
        return f"answer to {prompt}"

    r1 = ask("hello there")
    r2 = ask("world news")
    assert call_count[0] == 2
    assert r1 != r2


# ─── TTL Expiry ───────────────────────────────────────────────────────────────

def test_ttl_expiry():
    reset_stats()
    call_count = [0]

    @cache(threshold=0.92, embedder=_fake_embedder, ttl=1)
    def ask(prompt: str) -> str:
        call_count[0] += 1
        return f"answer to {prompt}"

    r1 = ask("ttl test")
    time.sleep(1.2)
    r2 = ask("ttl test")
    assert call_count[0] == 2
    assert r1 == r2


# ─── LRU Eviction ─────────────────────────────────────────────────────────────

def test_lru_eviction():
    reset_stats()
    store = MemoryStore()
    sc = SemantiCache(threshold=0.92, embedder=_fake_embedder, max_size=3, backend="memory")
    sc.store = store

    sc.get_or_call("a", lambda: "A")
    sc.get_or_call("b", lambda: "B")
    sc.get_or_call("c", lambda: "C")
    sc.get_or_call("d", lambda: "D")

    assert store.size() == 3
    # 'a' should have been evicted (oldest)
    hits = [store.find_similar(_fake_embedder(x), 0.92) for x in ["a", "b", "c", "d"]]
    assert hits[0] is None  # 'a' evicted
    assert hits[1] is not None
    assert hits[2] is not None
    assert hits[3] is not None


# ─── SQLite Persistence ───────────────────────────────────────────────────────

def test_sqlite_persists_across_restart():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # First process
        store1 = SQLiteStore(path=db_path)
        emb = _fake_embedder("persist me")
        store1.add("persist me", emb, "persisted response")
        store1.close()

        # Second process (new connection)
        store2 = SQLiteStore(path=db_path)
        result = store2.find_similar(emb, threshold=0.92)
        assert result is not None
        assert result.response == "persisted response"
        store2.close()
    finally:
        os.unlink(db_path)


# ─── Async Decorator ──────────────────────────────────────────────────────────

def test_async_decorator():
    reset_stats()
    call_count = [0]

    @cache(threshold=0.92, embedder=_fake_embedder)
    async def ask_async(prompt: str) -> str:
        call_count[0] += 1
        return f"async answer to {prompt}"

    r1 = asyncio.run(ask_async("async test"))
    r2 = asyncio.run(ask_async("async test"))
    assert call_count[0] == 1
    assert r1 == r2 == "async answer to async test"


# ─── Custom Embedder Injection ────────────────────────────────────────────────

def test_custom_embedder_injection():
    reset_stats()
    injected = [False]

    def custom_embedder(text: str) -> np.ndarray:
        injected[0] = True
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = 1.0
        return vec

    @cache(threshold=0.92, embedder=custom_embedder)
    def ask(prompt: str) -> str:
        return "ok"

    ask("test")
    assert injected[0]


# ─── SemantiCache Direct Usage ────────────────────────────────────────────────

def test_semanticache_direct_usage():
    reset_stats()
    sc = SemantiCache(threshold=0.92, embedder=_fake_embedder, backend="memory")
    r1 = sc.get_or_call("direct", lambda: "direct response")
    r2 = sc.get_or_call("direct", lambda: "should not call")
    assert r1 == r2 == "direct response"


# ─── FastAPI Endpoint Tests ───────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from semanticache.demo.server import app, demo_cache
    # Reset demo state
    reset_stats()
    demo_cache.store = MemoryStore()
    return TestClient(app)


def test_endpoint_ask_miss(client):
    res = client.post("/ask", json={"prompt": "endpoint test"})
    assert res.status_code == 200
    data = res.json()
    assert data["cache_hit"] is False
    assert "response" in data
    assert data["latency_ms"] > 0


def test_endpoint_ask_hit(client):
    client.post("/ask", json={"prompt": "hit test"})
    res = client.post("/ask", json={"prompt": "hit test"})
    assert res.status_code == 200
    data = res.json()
    assert data["cache_hit"] is True
    assert data["similarity_score"] == 1.0


def test_endpoint_ask_semantic_hit(client):
    # Use near-match embedder for this test
    from semanticache.demo.server import demo_cache
    demo_cache.embedder = _near_match_embedder
    demo_cache.store = MemoryStore()
    reset_stats()

    r1 = client.post("/ask", json={"prompt": "hello there"})
    assert r1.status_code == 200
    assert r1.json()["cache_hit"] is False

    r2 = client.post("/ask", json={"prompt": "hello everyone"})
    assert r2.status_code == 200
    data = r2.json()
    assert data["cache_hit"] is True
    assert data["similarity_score"] is not None
    assert data["similarity_score"] >= 0.92


def test_endpoint_stats(client):
    client.post("/ask", json={"prompt": "stats one"})
    client.post("/ask", json={"prompt": "stats one"})
    res = client.get("/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_calls"] >= 2
    assert data["cache_hits"] >= 1


def test_endpoint_entries(client):
    client.post("/ask", json={"prompt": "entries test"})
    res = client.get("/entries")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "prompt" in data[0]
    assert "response" in data[0]
