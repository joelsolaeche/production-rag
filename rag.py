"""
RAG Pipeline Core — Step 1

This module is the heart of the system. It takes a user query + retrieved
context chunks and produces a cited answer via OpenAI (GPT-4o-mini).

Key concepts:
- System prompt: constrains the model to only use provided context
- build_context: formats chunks into numbered [Source N] blocks
- generate_answer: orchestrates the call to OpenAI and tracks metrics
"""

import time
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# --------------------------------------------------------------------------- #
# OpenAI configuration — uses lazy initialization
# --------------------------------------------------------------------------- #
# The system prompt is the "contract" with the model. It tells the LLM:
#   1. Only answer from the provided context
#   2. Cite sources using [Source N] notation
#   3. Admit when information is insufficient
# This prevents hallucination — the #1 problem in production RAG.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Only use information from the provided context to answer questions.
2. Cite your sources using [Source N] notation, where N matches the source number in the context.
3. If the context does not contain enough information to answer the question, say:
   "I don't have enough information in the provided sources to answer this question."
4. Do not make up or infer information beyond what is explicitly stated in the sources.
5. Be concise and direct in your answers.
"""

# Lazy init — the client is created on first use, AFTER main.py loads .env
_client = None


def _get_client() -> OpenAI:
    """Initialize OpenAI client on first call (after dotenv has loaded the API key)."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """One retrieved chunk that was used as context."""
    index: int
    text_preview: str  # truncated to ~200 chars
    metadata: dict


@dataclass
class RAGResponse:
    """
    The structured output from a RAG query.

    - answer: the generated text with [Source N] citations
    - sources: list of Source objects showing what context was provided
    - latency_ms: how long the LLM call took
    - cache_hit: whether this came from the semantic cache
    """
    answer: str
    sources: list[Source]
    latency_ms: float
    cache_hit: bool = False


@dataclass
class RAGMetrics:
    """
    Module-level singleton that accumulates performance metrics.

    Why a singleton? Because in production you want ONE place to look at
    system health. The /metrics endpoint reads from this object.
    """
    total_queries: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per query. Returns 0 if no queries yet."""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of queries served from cache."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries


# The singleton — every part of the app imports this same instance
metrics = RAGMetrics()


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #
def build_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a numbered context block for the prompt.

    Each chunk is expected to have:
      - "text": the chunk content
      - "metadata": dict with at least a "title" key

    The output looks like:
        [Source 1] (Title Here)
        The actual text content...

        [Source 2] (Another Title)
        More text content...

    Why number them? So the model can cite specific sources in its answer,
    and the user can trace claims back to their origin.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("metadata", {}).get("title", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Source {i}] ({title})\n{text}")
    return "\n\n".join(context_parts)


# --------------------------------------------------------------------------- #
# Main generation function
# --------------------------------------------------------------------------- #
async def generate_answer(
    query: str,
    chunks: list[dict],
    use_cache: bool = True,
) -> RAGResponse:
    """
    The main RAG function. Takes a query and context chunks, returns a
    cited answer from OpenAI.

    Flow:
    1. Build the context string from chunks
    2. Construct the messages (system + user)
    3. Call OpenAI (gpt-4o-mini)
    4. Measure latency
    5. Update metrics
    6. Return structured RAGResponse

    Args:
        query: The user's question
        chunks: List of dicts with "text" and "metadata" keys
        use_cache: Flag for the caller (cache logic lives in main.py)
    """
    try:
        # Build the context string from our retrieved chunks
        context = build_context(chunks)

        # The user message includes BOTH the context and the question.
        # This is the standard RAG prompt pattern.
        user_message = f"""Context:
{context}

Question: {query}"""

        # Time the OpenAI call — latency is a key production metric
        start = time.time()

        # OpenAI uses a messages array with roles (system, user, assistant)
        # instead of a single string. The system message sets behavior,
        # the user message is the actual query.
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        latency_ms = (time.time() - start) * 1000

        # Extract the answer text from OpenAI's response structure
        answer_text = response.choices[0].message.content

        # Build the source list with truncated previews
        sources = []
        for i, chunk in enumerate(chunks, 1):
            sources.append(Source(
                index=i,
                text_preview=chunk.get("text", "")[:200],
                metadata=chunk.get("metadata", {}),
            ))

        # Update the global metrics tracker
        metrics.total_queries += 1
        metrics.total_latency_ms += latency_ms

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            latency_ms=latency_ms,
            cache_hit=False,
        )

    except Exception as e:
        # Always count errors BEFORE re-raising — if we re-raise first,
        # the counter never increments and we lose visibility into failures.
        metrics.errors += 1
        raise
