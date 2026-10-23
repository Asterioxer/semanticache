"""semanticache: lightweight semantic caching for LLM API calls."""

from .cache import cache, SemantiCache
from .stats import get_stats

__all__ = ["cache", "SemantiCache", "get_stats"]
__version__ = "0.1.0"
