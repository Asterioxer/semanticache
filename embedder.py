"""Lazy-loaded default embedder using sentence-transformers."""

from typing import Callable
import numpy as np

# Module-level singleton for the default model — lazy-loaded on first use.
_default_model = None


def _load_default_model():
    """Lazy-load the all-MiniLM-L6-v2 model."""
    global _default_model
    if _default_model is None:
        # all-MiniLM-L6-v2 is ~80MB, fast, and high quality for sentence similarity.
        # We import here (not at module top) to keep `import semanticache` fast.
        from sentence_transformers import SentenceTransformer
        _default_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _default_model


def default_embedder(text: str) -> np.ndarray:
    """Return a 384-dim numpy array for the given text."""
    model = _load_default_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding


def get_default_embedder() -> Callable[[str], np.ndarray]:
    """Return the default embedding function."""
    return default_embedder
