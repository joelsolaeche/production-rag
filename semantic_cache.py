"""
Semantic Cache — Step 3

WHY a semantic cache?
A normal cache (like Redis) matches on EXACT strings:
  "What is RAG?" → cache hit
  "what is rag?" → cache MISS (different string!)
  "Explain RAG to me" → cache MISS (totally different string!)

A semantic cache matches on MEANING using embeddings:
  "What is RAG?" → embed → [0.12, 0.87, 0.34, ...]  → cache hit
  "Explain RAG to me" → embed → [0.11, 0.85, 0.36, ...]  → cache HIT! (similar vector)

This saves Gemini API calls ($$) and reduces latency for repeated/similar questions.

HOW it works — Cosine Similarity:
Embeddings are vectors (lists of numbers) that represent meaning.
Cosine similarity measures the angle between two vectors:

                    A · B           (dot product)
  cos(θ) = ─────────────────── = ──────────────────────
             ||A|| × ||B||       (magnitude × magnitude)

  - Result of 1.0 → identical direction → same meaning
  - Result of 0.0 → perpendicular → unrelated
  - Result of -1.0 → opposite → opposite meaning

We set a threshold (default 0.95) — if similarity >= threshold, it's a cache hit.

Visual intuition:
  "What is RAG?"       →  vector pointing ↗ this way
  "Explain RAG to me"  →  vector pointing ↗ almost same way (cos = 0.96) → HIT
  "How to cook pasta?" →  vector pointing ↘ different way (cos = 0.23) → MISS
"""

import time
import math
from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Cache Entry
# --------------------------------------------------------------------------- #
@dataclass
class CacheEntry:
    """
    One cached response.

    We store the original query (for debugging), its embedding (for similarity
    comparison), the full response, and TTL info for expiration.

    TTL (Time To Live) = how long this entry is valid. After ttl_seconds,
    the entry is considered stale and will be evicted. Why? Because the
    underlying data might have changed — you don't want to serve an answer
    about "current pricing" that's 3 hours old.
    """
    query: str
    embedding: list[float]
    response: dict
    created_at: float  # time.time() when stored
    ttl_seconds: int = 3600  # default: 1 hour

    def is_expired(self) -> bool:
        """Check if this entry has lived past its TTL."""
        return (time.time() - self.created_at) > self.ttl_seconds


# --------------------------------------------------------------------------- #
# Cosine Similarity
# --------------------------------------------------------------------------- #
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Step by step:
      1. dot_product = sum(a[i] * b[i] for all i)  — how aligned are they?
      2. norm_a = sqrt(sum(a[i]² for all i))        — length of vector A
      3. norm_b = sqrt(sum(b[i]² for all i))        — length of vector B
      4. similarity = dot_product / (norm_a * norm_b)

    We compute all three in a SINGLE pass through both vectors for efficiency.
    In production, this runs on every cache entry for every query, so speed matters.

    Edge case: if either vector has zero length (all zeros), return 0.0
    to avoid division by zero.
    """
    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    # Single pass — compute dot product and both norms simultaneously
    for x, y in zip(a, b):
        dot_product += x * y
        norm_a_sq += x * x
        norm_b_sq += y * y

    # Handle zero vectors (e.g., empty embedding or all-zeros)
    if norm_a_sq == 0.0 or norm_b_sq == 0.0:
        return 0.0

    return dot_product / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))


# --------------------------------------------------------------------------- #
# The Semantic Cache
# --------------------------------------------------------------------------- #
class SemanticCache:
    """
    A cache that matches queries by meaning, not exact string.

    How a lookup works:
      1. Receive a query embedding
      2. Remove any expired entries (past their TTL)
      3. Compare the query embedding against EVERY cached entry using cosine similarity
      4. If the best match is above the threshold → return cached response
      5. Otherwise → return None (cache miss)

    Why scan all entries? With typical cache sizes (< 1000), a linear scan
    over float arrays is extremely fast. A more complex index (like HNSW)
    would only help at 100K+ entries.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
    ):
        # 0.95 is conservative — means queries must be ~95% similar in meaning.
        # Lower (0.90) = more cache hits but risk of wrong answers.
        # Higher (0.98) = safer but caches less.
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []

    def get(self, query_embedding: list[float]) -> Optional[dict]:
        """
        Look up a cached response by semantic similarity.

        Returns the cached response dict if a match is found, None otherwise.

        Algorithm:
          1. Evict expired entries
          2. Scan all remaining entries, compute cosine similarity
          3. Track the best (highest similarity) match
          4. Return it only if it exceeds the threshold
        """
        # Step 1: Evict expired entries
        self.entries = [e for e in self.entries if not e.is_expired()]

        # Step 2-3: Find the best matching entry
        best_match: Optional[CacheEntry] = None
        best_similarity = -1.0  # Start below any possible similarity

        for entry in self.entries:
            sim = cosine_similarity(query_embedding, entry.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_match = entry

        # Step 4: Only return if similarity is above threshold
        if best_match is not None and best_similarity >= self.similarity_threshold:
            return best_match.response

        return None

    def put(
        self,
        query: str,
        query_embedding: list[float],
        response: dict,
        ttl_seconds: int = 3600,
    ) -> None:
        """
        Store a new response in the cache.

        If the cache is full, evict the oldest 25% of entries first.
        Why 25%? Evicting one-at-a-time means you're evicting on almost
        every insert when at capacity. Evicting 25% gives you breathing
        room before the next eviction, reducing overhead.
        """
        # Evict oldest 25% if at capacity
        if len(self.entries) >= self.max_entries:
            # Sort by creation time, remove the oldest quarter
            self.entries.sort(key=lambda e: e.created_at)
            cutoff = len(self.entries) // 4  # 25% of entries
            self.entries = self.entries[cutoff:]

        # Add the new entry
        self.entries.append(CacheEntry(
            query=query,
            embedding=query_embedding,
            response=response,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
        ))

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self.entries)


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #
semantic_cache = SemanticCache()
