"""Benchmark script for semanticache."""

import time

from semanticache import cache
from semanticache.stats import reset_stats


def fake_embedder(text: str):
    import numpy as np

    np.random.seed(hash(text) & 0xFFFFFFFF)
    vec = np.random.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def mock_llm(prompt: str) -> str:
    time.sleep(0.2)
    return f"Response to: {prompt}"


@cache(threshold=0.92, embedder=fake_embedder, cost_per_call=0.02)
def ask(prompt: str) -> str:
    return mock_llm(prompt)


PROMPTS = [
    "What is the capital of France?",
    "Tell me the capital city of France",
    "France capital city",
    "What is the capital of France?",  # exact repeat
    "How do I write a Python class?",
    "How to create a class in Python",
    "Python class syntax example",
    "How do I write a Python class?",  # exact repeat
    "Explain quantum computing simply",
    "Quantum computing for beginners",
    "Explain quantum computing simply",  # exact repeat
    "What are the benefits of exercise?",
    "Why should I exercise regularly?",
    "Benefits of physical activity",
    "What are the benefits of exercise?",  # exact repeat
    "How to make sourdough bread",
    "Sourdough bread recipe",
    "Making sourdough at home",
    "How to make sourdough bread",  # exact repeat
]


def main():
    reset_stats()
    print("Running semanticache benchmark...")
    print(f"Total prompts: {len(PROMPTS)}")
    print("-" * 40)

    start = time.perf_counter()
    for p in PROMPTS:
        t0 = time.perf_counter()
        response = ask(p)
        latency = (time.perf_counter() - t0) * 1000
        print(f"[{latency:6.1f}ms] {p[:50]:<50}")
    total_time = (time.perf_counter() - start) * 1000

    from semanticache import get_stats

    stats = get_stats()

    print("-" * 40)
    print(f"Total time:        {total_time:.1f} ms")
    print(f"Cache hit rate:    {stats['hit_rate_percent']:.1f}%")
    print(f"Cache hits:        {stats['cache_hits']}")
    print(f"Cache misses:      {stats['cache_misses']}")
    print(f"Estimated cost saved: ${stats['estimated_cost_saved']:.4f}")
    print(f"Estimated time saved: {stats['estimated_time_saved_ms'] / 1000:.2f}s")
    print("-" * 40)


if __name__ == "__main__":
    main()
