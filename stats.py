"""In-process statistics tracking for semanticache."""

from dataclasses import dataclass, field
from typing import Optional
import threading


@dataclass
class Stats:
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avoided_calls: int = 0
    estimated_cost_saved: float = 0.0
    estimated_time_saved_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_hit(self, cost_per_call: Optional[float] = None, latency_ms: Optional[float] = None):
        with self._lock:
            self.total_calls += 1
            self.cache_hits += 1
            self.avoided_calls += 1
            if cost_per_call is not None:
                self.estimated_cost_saved += cost_per_call
            if latency_ms is not None:
                self.estimated_time_saved_ms += latency_ms

    def record_miss(self, cost_per_call: Optional[float] = None, latency_ms: Optional[float] = None):
        with self._lock:
            self.total_calls += 1
            self.cache_misses += 1

    def to_dict(self) -> dict:
        with self._lock:
            total = self.total_calls
            hit_rate = (self.cache_hits / total * 100) if total > 0 else 0.0
            return {
                "total_calls": self.total_calls,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "avoided_calls": self.avoided_calls,
                "hit_rate_percent": round(hit_rate, 2),
                "estimated_cost_saved": round(self.estimated_cost_saved, 4),
                "estimated_time_saved_ms": round(self.estimated_time_saved_ms, 2),
            }

    def reset(self):
        """Reset all counters in-place (keeps object identity)."""
        with self._lock:
            self.total_calls = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.avoided_calls = 0
            self.estimated_cost_saved = 0.0
            self.estimated_time_saved_ms = 0.0


# Global singleton stats instance
_global_stats = Stats()


def get_stats() -> dict:
    """Return current statistics as a plain dict."""
    return _global_stats.to_dict()


def reset_stats():
    """Reset all statistics to zero. Primarily for testing."""
    _global_stats.reset()
