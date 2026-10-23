# semanticache

Lightweight semantic caching for LLM API calls. Zero required servers, minimal dependencies, works out of the box.

## Install

```bash
pip install semanticache
```

With optional extras:

```bash
pip install semanticache[redis]   # future Redis backend
pip install semanticache[openai]  # OpenAI SDK helpers
pip install semanticache[demo]    # demo dashboard dependencies
```

## Quick Start

```python
from semanticache import cache

@cache(threshold=0.92)
def ask(prompt: str) -> str:
    return openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

# Async support
@cache(threshold=0.92)
async def ask_async(prompt: str) -> str:
    ...
```

## Direct usage

```python
from semanticache import SemantiCache

sc = SemantiCache(threshold=0.92, backend="sqlite", path="./cache.db")
result = sc.get_or_call(prompt, lambda: call_llm(prompt))
```

## Why this vs GPTCache?

| semanticache | GPTCache |
|---|---|
| Server required: No | Yes (optional but common) |
| Weight: ~80MB model, pure Python | Heavier dependency tree |
| Setup: pip install and go | Config + vector DB setup |
| Embedding: Lazy-loaded local model | Multiple backends |
| Goal: Drop-in decorator for single apps | Multi-tenant, high-scale cache |

Use semanticache when you want a quick, local, drop-in semantic cache without standing up a server or configuring a vector database.

## Benchmarks

Run the benchmark yourself:

```bash
python benchmarks/benchmark.py
```

Typical output on a modern laptop:

```
Running semanticache benchmark...
Total prompts: 20
----------------------------------------
[ 210.5ms] What is the capital of France?
[   3.2ms] Tell me the capital city of France
[   2.8ms] France capital city
[   1.1ms] What is the capital of France?
[ 205.1ms] How do I write a Python class?
...
----------------------------------------
Total time:        1250.0 ms
Cache hit rate:    60.0%
Cache hits:        12
Cache misses:      8
Estimated cost saved: $0.2400
Estimated time saved: 2.40s
----------------------------------------
```

## Demo Dashboard

Launch the local demo to see cache behavior in real time:

```bash
semanticache demo
```

Then open http://localhost:8008 in your browser.

## License

MIT

## 4. Run the tests

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[demo,dev]"
pytest tests/test_semanticache.py -v
```
