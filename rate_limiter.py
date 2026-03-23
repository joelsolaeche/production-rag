"""
Rate Limiter — Step 2

WHY rate limiting?
In production, without rate limiting:
  - A single user could spam your API and rack up huge Gemini bills
  - A bug in a client could send 1000 requests/second
  - You have no way to guarantee fair access across users

HOW it works — Sliding Window Algorithm:
Instead of resetting a counter every minute (fixed window), we store the
TIMESTAMP of each request. To check if a client is over the limit, we:
  1. Remove all timestamps older than the window (e.g., 60 seconds)
  2. Count what's left
  3. If count >= limit → reject with HTTP 429

This is "sliding" because the window moves with the current time, not
fixed to clock boundaries. It's smoother and harder to game.

Example with limit=3 per minute:
  t=0s   → request → timestamps: [0]          → allowed (1 < 3)
  t=10s  → request → timestamps: [0, 10]      → allowed (2 < 3)
  t=20s  → request → timestamps: [0, 10, 20]  → allowed (3 = 3... just made it)
  t=30s  → request → timestamps: [0, 10, 20]  → REJECTED (3 >= 3)
  t=61s  → request → timestamps: [10, 20, 61] → allowed (t=0 expired!)
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from fastapi import HTTPException


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class RateLimitConfig:
    """
    Three separate limits to control different abuse patterns:
    - per_minute: stops burst abuse (bot hammering the API)
    - per_hour: stops sustained abuse (someone scripting all day)
    - tokens_per_minute: stops cost abuse (few requests but huge prompts)
    """
    requests_per_minute: int = 20
    requests_per_hour: int = 200
    tokens_per_minute: int = 100_000


# --------------------------------------------------------------------------- #
# The Rate Limiter
# --------------------------------------------------------------------------- #
class InMemoryRateLimiter:
    """
    Sliding window rate limiter that tracks per-client usage.

    Internal storage:
      minute_windows: { "client_id": [timestamp, timestamp, ...] }
      hour_windows:   { "client_id": [timestamp, timestamp, ...] }
      token_usage:    { "client_id": [(timestamp, count), ...] }

    We use defaultdict(list) so we never need to check if a client exists —
    accessing a new client_id automatically creates an empty list.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        # defaultdict(list) → accessing a missing key returns [] automatically
        self.minute_windows: dict[str, list[float]] = defaultdict(list)
        self.hour_windows: dict[str, list[float]] = defaultdict(list)
        # Token tracking stores (timestamp, token_count) tuples because
        # each request uses a different number of tokens
        self.token_usage: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def _clean_window(self, window: list[float], max_age_seconds: float) -> list[float]:
        """
        Remove expired entries from a timestamp list.

        This is the core of the sliding window — we discard anything older
        than `max_age_seconds` from now. What remains is the "active" window.
        """
        cutoff = time.time() - max_age_seconds
        # List comprehension filters out old timestamps
        return [t for t in window if t > cutoff]

    def check_rate_limit(self, client_id: str) -> None:
        """
        Check if a client is allowed to make a request.

        Steps:
        1. Clean expired entries from both windows
        2. Check minute limit
        3. Check hour limit
        4. If both pass, record the current timestamp

        Raises HTTPException(429) if any limit is exceeded.
        Why 429? It's the HTTP standard for "Too Many Requests" — clients
        and load balancers understand this code and can back off automatically.
        """
        now = time.time()

        # Step 1: Clean expired entries
        self.minute_windows[client_id] = self._clean_window(
            self.minute_windows[client_id], 60  # 60 seconds = 1 minute
        )
        self.hour_windows[client_id] = self._clean_window(
            self.hour_windows[client_id], 3600  # 3600 seconds = 1 hour
        )

        # Step 2: Check per-minute limit
        if len(self.minute_windows[client_id]) >= self.config.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.config.requests_per_minute} requests per minute"
            )

        # Step 3: Check per-hour limit
        if len(self.hour_windows[client_id]) >= self.config.requests_per_hour:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.config.requests_per_hour} requests per hour"
            )

        # Step 4: Both checks passed — record this request
        self.minute_windows[client_id].append(now)
        self.hour_windows[client_id].append(now)

    def record_tokens(self, client_id: str, token_count: int) -> None:
        """
        Track token usage for a client. Called AFTER a successful Gemini response.

        Why track tokens separately from requests?
        Because one request with a huge context could cost more than 100 small ones.
        Token tracking lets you set cost-based limits, not just request-based.
        """
        now = time.time()
        self.token_usage[client_id].append((now, token_count))

    def get_token_usage(self, client_id: str) -> int:
        """
        Get the total tokens used by a client in the current minute window.

        Cleans expired entries first, then sums up the token counts.
        """
        cutoff = time.time() - 60
        # Filter to only entries within the last minute
        self.token_usage[client_id] = [
            (t, count) for t, count in self.token_usage[client_id]
            if t > cutoff
        ]
        # Sum the token counts from all remaining entries
        return sum(count for _, count in self.token_usage[client_id])


# --------------------------------------------------------------------------- #
# Module-level singleton — the whole app shares one limiter instance
# --------------------------------------------------------------------------- #
rate_limiter = InMemoryRateLimiter()
