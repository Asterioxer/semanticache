"""Storage backends for semanticache: in-memory and SQLite."""

import abc
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CacheEntry:
    prompt: str
    embedding: np.ndarray
    response: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class BaseStore(abc.ABC):
    """Abstract base for semanticache storage backends."""

    @abc.abstractmethod
    def get_by_hash(self, prompt_hash: str) -> Optional[CacheEntry]:
        """Exact-match lookup by prompt hash (fast path, no embedding needed)."""
        ...

    @abc.abstractmethod
    def find_similar(
        self, embedding: np.ndarray, threshold: float, ttl: Optional[int] = None
    ) -> Optional[CacheEntry]:
        """Return the most similar cached entry above threshold, or None."""
        ...

    @abc.abstractmethod
    def add(
        self,
        prompt: str,
        embedding: np.ndarray,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def evict_expired(self, ttl: int) -> int:
        """Remove entries older than ttl seconds. Return count removed."""
        ...

    @abc.abstractmethod
    def evict_oldest(self, n: int) -> int:
        """Remove the n oldest entries (LRU). Return count removed."""
        ...

    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return recent entries as plain dicts for display."""
        ...

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


class MemoryStore(BaseStore):
    """Pure in-memory store with brute-force cosine similarity."""

    def __init__(self):
        self._entries: List[CacheEntry] = []
        self._exact: Dict[str, int] = {}  # prompt_hash -> index in _entries
        self._lock = threading.RLock()

    def get_by_hash(self, prompt_hash: str) -> Optional[CacheEntry]:
        with self._lock:
            idx = self._exact.get(prompt_hash)
            if idx is None:
                return None
            entry = self._entries[idx]
            # Update access time for LRU
            now = time.time()
            self._entries[idx] = CacheEntry(
                prompt=entry.prompt,
                embedding=entry.embedding,
                response=entry.response,
                timestamp=entry.timestamp,
                metadata={**(entry.metadata or {}), "_last_access": now},
            )
            return entry

    def find_similar(
        self, embedding: np.ndarray, threshold: float, ttl: Optional[int] = None
    ) -> Optional[CacheEntry]:
        with self._lock:
            now = time.time()
            best_score = -1.0
            best_entry = None
            best_idx = -1

            for idx, entry in enumerate(self._entries):
                if ttl is not None and (now - entry.timestamp) > ttl:
                    continue
                score = self._cosine_similarity(embedding, entry.embedding)
                if score >= threshold and score > best_score:
                    best_score = score
                    best_entry = entry
                    best_idx = idx

            if best_idx >= 0:
                now = time.time()
                self._entries[best_idx] = CacheEntry(
                    prompt=best_entry.prompt,
                    embedding=best_entry.embedding,
                    response=best_entry.response,
                    timestamp=best_entry.timestamp,
                    metadata={**(best_entry.metadata or {}), "_last_access": now},
                )

            return best_entry

    def add(
        self,
        prompt: str,
        embedding: np.ndarray,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            ph = self._prompt_hash(prompt)
            now = time.time()
            entry = CacheEntry(
                prompt=prompt,
                embedding=embedding,
                response=response,
                timestamp=now,
                metadata={**(metadata or {}), "_last_access": now},
            )
            if ph in self._exact:
                self._entries[self._exact[ph]] = entry
            else:
                self._entries.append(entry)
                self._exact[ph] = len(self._entries) - 1

    def evict_expired(self, ttl: int) -> int:
        with self._lock:
            now = time.time()
            new_entries = []
            removed = 0
            for entry in self._entries:
                if (now - entry.timestamp) > ttl:
                    removed += 1
                else:
                    new_entries.append(entry)
            self._entries = new_entries
            self._exact = {
                self._prompt_hash(e.prompt): i for i, e in enumerate(self._entries)
            }
            return removed

    def evict_oldest(self, n: int) -> int:
        with self._lock:
            if n <= 0:
                return 0
            sorted_entries = sorted(
                enumerate(self._entries),
                key=lambda x: (x[1].metadata or {}).get("_last_access", x[1].timestamp),
            )
            to_remove = set(idx for idx, _ in sorted_entries[:n])
            new_entries = [e for i, e in enumerate(self._entries) if i not in to_remove]
            removed = len(self._entries) - len(new_entries)
            self._entries = new_entries
            self._exact = {
                self._prompt_hash(e.prompt): i for i, e in enumerate(self._entries)
            }
            return removed

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            result = []
            for entry in reversed(self._entries):
                result.append({
                    "prompt": entry.prompt,
                    "response": entry.response,
                    "timestamp": entry.timestamp,
                })
                if limit and len(result) >= limit:
                    break
            return result


class SQLiteStore(BaseStore):
    """SQLite-backed store. Loads embeddings into RAM for fast search."""

    def __init__(self, path: str = "./cache.db"):
        self._path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._init_db()
        self._load_into_memory()

    def close(self):
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()

    def _init_db(self):
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT UNIQUE NOT NULL,
                    prompt TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    response TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_time REAL NOT NULL,
                    metadata TEXT
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(prompt_hash)
            """)

    def _load_into_memory(self):
        self._entries: List[CacheEntry] = []
        self._exact: Dict[str, int] = {}
        cursor = self._conn.execute(
            "SELECT prompt_hash, prompt, embedding, response, timestamp, access_time, metadata FROM cache"
        )
        for row in cursor:
            ph, prompt, emb_blob, response, timestamp, access_time, meta_json = row
            embedding = np.frombuffer(emb_blob, dtype=np.float32)
            metadata = json.loads(meta_json) if meta_json else {}
            metadata["_last_access"] = access_time
            entry = CacheEntry(
                prompt=prompt,
                embedding=embedding,
                response=response,
                timestamp=timestamp,
                metadata=metadata,
            )
            self._exact[ph] = len(self._entries)
            self._entries.append(entry)

    def get_by_hash(self, prompt_hash: str) -> Optional[CacheEntry]:
        with self._lock:
            idx = self._exact.get(prompt_hash)
            if idx is None:
                return None
            entry = self._entries[idx]
            now = time.time()
            self._update_access_time(idx, now)
            return entry

    def find_similar(
        self, embedding: np.ndarray, threshold: float, ttl: Optional[int] = None
    ) -> Optional[CacheEntry]:
        with self._lock:
            now = time.time()
            best_score = -1.0
            best_entry = None
            best_idx = -1

            for idx, entry in enumerate(self._entries):
                if ttl is not None and (now - entry.timestamp) > ttl:
                    continue
                score = self._cosine_similarity(embedding, entry.embedding)
                if score >= threshold and score > best_score:
                    best_score = score
                    best_entry = entry
                    best_idx = idx

            if best_idx >= 0:
                self._update_access_time(best_idx, now)

            return best_entry

    def _update_access_time(self, idx: int, access_time: float):
        entry = self._entries[idx]
        ph = self._prompt_hash(entry.prompt)
        entry.metadata["_last_access"] = access_time
        with self._conn:
            self._conn.execute(
                "UPDATE cache SET access_time = ? WHERE prompt_hash = ?",
                (access_time, ph),
            )

    def add(
        self,
        prompt: str,
        embedding: np.ndarray,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            ph = self._prompt_hash(prompt)
            now = time.time()
            emb_blob = embedding.astype(np.float32).tobytes()
            meta_json = json.dumps(metadata) if metadata else None

            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO cache (prompt_hash, prompt, embedding, response, timestamp, access_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(prompt_hash) DO UPDATE SET
                        prompt=excluded.prompt,
                        embedding=excluded.embedding,
                        response=excluded.response,
                        timestamp=excluded.timestamp,
                        access_time=excluded.access_time,
                        metadata=excluded.metadata
                    """,
                    (ph, prompt, emb_blob, response, now, now, meta_json),
                )

            entry = CacheEntry(
                prompt=prompt,
                embedding=embedding,
                response=response,
                timestamp=now,
                metadata={**(metadata or {}), "_last_access": now},
            )
            if ph in self._exact:
                self._entries[self._exact[ph]] = entry
            else:
                self._exact[ph] = len(self._entries)
                self._entries.append(entry)

    def evict_expired(self, ttl: int) -> int:
        with self._lock:
            cutoff = time.time() - ttl
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM cache WHERE timestamp < ?", (cutoff,)
                )
                removed = cursor.rowcount
            self._load_into_memory()
            return removed

    def evict_oldest(self, n: int) -> int:
        with self._lock:
            if n <= 0:
                return 0
            with self._conn:
                cursor = self._conn.execute(
                    """
                    DELETE FROM cache
                    WHERE id IN (
                        SELECT id FROM cache ORDER BY access_time ASC LIMIT ?
                    )
                    """,
                    (n,),
                )
                removed = cursor.rowcount
            self._load_into_memory()
            return removed

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            result = []
            for entry in reversed(self._entries):
                result.append({
                    "prompt": entry.prompt,
                    "response": entry.response,
                    "timestamp": entry.timestamp,
                })
                if limit and len(result) >= limit:
                    break
            return result
