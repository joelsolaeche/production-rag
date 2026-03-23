"""
Production RAG API — Step 5

This is where everything comes together. Each module we built is a
"building block" and this file is the "assembly":

  rag.py            → generate_answer(), metrics
  rate_limiter.py   → rate_limiter singleton
  semantic_cache.py → semantic_cache singleton
  evaluation.py     → run_evaluation()

The API has 5 endpoints:
  POST /query    → the main RAG pipeline (rate limit → cache → retrieve → generate)
  POST /index    → add documents to the knowledge base
  POST /evaluate → run the evaluation suite
  GET  /metrics  → system health dashboard
  GET  /health   → simple liveness check

The /query endpoint is the most interesting — it chains every module
together in a specific order:

  Request arrives
    │
    ├─ 1. Rate limit check     → 429 if exceeded
    ├─ 2. Embed the query      → convert text to vector
    ├─ 3. Check semantic cache  → return early if cache hit
    ├─ 4. Retrieve chunks      → hybrid search for relevant docs
    ├─ 5. Generate answer      → call Gemini with context
    ├─ 6. Store in cache       → save for future similar queries
    ├─ 7. Record token usage   → for token-based rate limiting
    │
    └─ Return response
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env file BEFORE any module that reads environment variables.
# This makes os.environ.get("GOOGLE_API_KEY") work in rag.py and evaluation.py.
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from rag import generate_answer, metrics as rag_metrics, build_context
from rate_limiter import rate_limiter
from semantic_cache import semantic_cache
from evaluation import run_evaluation

# --------------------------------------------------------------------------- #
# Embedding client — uses OpenAI's embedding API instead of local model
# --------------------------------------------------------------------------- #
# Why OpenAI API instead of local sentence-transformers?
#   - sentence-transformers + torch = ~2GB, won't fit on Render free tier (512MB)
#   - OpenAI text-embedding-3-small is fast, cheap, and produces 1536-dim vectors
#   - No GPU needed, no model download, tiny deployment footprint
from openai import OpenAI

_embed_client = None


def _get_embed_client() -> OpenAI:
    """Lazy init for the embedding client."""
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _embed_client


app = FastAPI(
    title="Production RAG API",
    description="RAG system with evaluation, caching, and rate limiting",
)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def embed_query(query: str) -> list[float]:
    """
    Convert a text query into a 1536-dimensional embedding vector via OpenAI API.

    Uses text-embedding-3-small — cheapest OpenAI embedding model.
    This vector captures the MEANING of the query, not the exact words.
    "What is RAG?" and "Explain RAG" produce similar vectors.

    Used for:
      - Semantic cache lookups (find similar past queries)
      - Vector search in the retriever (find relevant documents)
    """
    response = _get_embed_client().embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    return response.data[0].embedding


async def retrieve(query: str, limit: int = 5) -> list[dict]:
    """
    Retrieve relevant document chunks for a query.

    In a full production system, this would call your vector database
    (Pinecone, Weaviate, etc.) with hybrid search (BM25 + semantic).

    For now, we return placeholder data so the API works end-to-end.
    Replace this with your Lab 04 retrieval when ready.

    Each chunk has:
      - id: unique identifier for evaluation tracking
      - text: the actual content
      - metadata: title and source info for citations
    """
    # TODO: Replace with real retrieval from your vector store
    # Example: results = pinecone_index.query(vector=embed_query(query), top_k=limit)
    return [
        {
            "id": "doc1_chunk1",
            "text": (
                "Hybrid search combines two retrieval strategies: BM25 (keyword-based) "
                "and semantic search (embedding-based). BM25 excels at exact keyword "
                "matching while semantic search captures meaning and synonyms. "
                "By combining both with Reciprocal Rank Fusion (RRF), you get the "
                "best of both worlds."
            ),
            "metadata": {"title": "Hybrid Search Overview", "source": "docs/search.md"},
        },
        {
            "id": "doc1_chunk2",
            "text": (
                "BM25 scoring works by computing term frequency (TF) and inverse "
                "document frequency (IDF) for each query term. Documents with rare "
                "query terms appearing frequently score highest. The algorithm also "
                "accounts for document length normalization."
            ),
            "metadata": {"title": "BM25 Scoring", "source": "docs/bm25.md"},
        },
        {
            "id": "doc1_chunk3",
            "text": (
                "Chunking strategies determine how documents are split before indexing. "
                "Common approaches include fixed-size chunks (e.g., 512 tokens), "
                "sentence-based splitting, and recursive character splitting. "
                "Overlap between chunks (e.g., 50 tokens) helps preserve context "
                "that spans chunk boundaries."
            ),
            "metadata": {"title": "Chunking Strategies", "source": "docs/chunking.md"},
        },
    ]


# --------------------------------------------------------------------------- #
# Request/Response models (Pydantic)
# --------------------------------------------------------------------------- #
# Pydantic models define the SHAPE of request/response JSON.
# FastAPI uses these to:
#   1. Validate incoming requests (reject bad data with clear errors)
#   2. Generate OpenAPI/Swagger docs automatically
#   3. Provide type hints for your IDE

class QueryRequest(BaseModel):
    query: str
    limit: int = 5           # how many chunks to retrieve
    use_cache: bool = True    # enable/disable semantic cache
    client_id: str = "default"  # for per-client rate limiting


class IndexRequest(BaseModel):
    doc_id: str
    title: str
    text: str


class EvalItem(BaseModel):
    query: str
    relevant_doc_ids: list[str]


class EvalRequest(BaseModel):
    dataset: list[EvalItem]
    k: int = 5


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health():
    """
    Liveness check. Load balancers and orchestrators (Railway, K8s) ping
    this to know if the service is alive. Keep it dead simple — no DB
    calls, no external dependencies.
    """
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    The main RAG pipeline. This is where all modules come together.

    The ORDER of operations matters:
      1. Rate limit FIRST — reject abusers before doing any work
      2. Embed the query — needed for cache lookup AND retrieval
      3. Cache check — avoid expensive Gemini call if possible
      4. Retrieve — find relevant document chunks
      5. Generate — call Gemini with context
      6. Cache store — save for future similar queries
      7. Record tokens — track usage for token-based limits
    """
    # Step 1: Rate limit — cheapest check, do it first
    rate_limiter.check_rate_limit(request.client_id)

    # Step 2: Embed the query (needed for both cache and retrieval)
    query_embedding = embed_query(request.query)

    # Step 3: Check semantic cache (if enabled)
    if request.use_cache:
        cached = semantic_cache.get(query_embedding)
        if cached is not None:
            # Cache hit! Update metrics and return immediately.
            # This skips retrieval AND generation — huge latency win.
            rag_metrics.total_queries += 1
            rag_metrics.cache_hits += 1
            return cached

    # Step 4: Cache miss — retrieve relevant chunks
    chunks = await retrieve(request.query, request.limit)

    # Step 5: Generate answer via Gemini
    rag_response = await generate_answer(
        query=request.query,
        chunks=chunks,
        use_cache=request.use_cache,
    )

    # Build the response dict
    response = {
        "answer": rag_response.answer,
        "sources": [
            {
                "index": s.index,
                "text_preview": s.text_preview,
                "metadata": s.metadata,
            }
            for s in rag_response.sources
        ],
        "latency_ms": rag_response.latency_ms,
        "cache_hit": False,
    }

    # Step 6: Store in semantic cache for future similar queries
    if request.use_cache:
        semantic_cache.put(
            query=request.query,
            query_embedding=query_embedding,
            response=response,
        )

    # Step 7: Record token usage (estimate — real count would come from API response)
    estimated_tokens = len(request.query.split()) * 2 + 500  # rough estimate
    rate_limiter.record_tokens(request.client_id, estimated_tokens)

    return response


@app.post("/index")
async def index_endpoint(request: IndexRequest):
    """
    Add a document to the knowledge base.

    In production, this would:
      1. Chunk the document
      2. Embed each chunk
      3. Store in vector DB (Pinecone, etc.)
      4. Update the BM25 index

    For now, returns a stub acknowledgement. Plug in your Lab 04
    indexing pipeline here.
    """
    # TODO: Replace with real indexing pipeline
    return {
        "status": "indexed",
        "doc_id": request.doc_id,
        "title": request.title,
        "chunks_created": max(1, len(request.text) // 500),  # estimate
    }


@app.post("/evaluate")
async def evaluate_endpoint(request: EvalRequest):
    """
    Run the evaluation suite over a dataset.

    This endpoint wires up the retrieval and generation functions as
    callbacks, then passes them to run_evaluation(). This decoupling
    means the evaluation module doesn't know or care about HOW retrieval
    or generation work — it just calls the functions.
    """
    # Define retrieval callback — wraps our retrieve() function
    async def retrieval_fn(query: str) -> list[dict]:
        return await retrieve(query, request.k)

    # Define generation callback — wraps our generate_answer() function
    async def generation_fn(query: str, chunks: list[dict]) -> str:
        response = await generate_answer(query, chunks)
        return response.answer

    # Run the full evaluation
    result = await run_evaluation(
        eval_dataset=[item.model_dump() for item in request.dataset],
        retrieval_fn=retrieval_fn,
        generation_fn=generation_fn,
        k=request.k,
    )

    return {
        "precision_at_k": result.precision_at_k,
        "mrr": result.mrr,
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "details": result.details,
    }


@app.get("/metrics")
async def metrics_endpoint():
    """
    System health dashboard. In production, a monitoring tool (Grafana,
    Datadog) would scrape this endpoint periodically.

    Returns:
      - rag: query counts, latency, cache hit rate, errors
      - cache: current size of the semantic cache
    """
    return {
        "rag": {
            "total_queries": rag_metrics.total_queries,
            "avg_latency_ms": rag_metrics.avg_latency_ms,
            "cache_hit_rate": rag_metrics.cache_hit_rate,
            "cache_hits": rag_metrics.cache_hits,
            "errors": rag_metrics.errors,
        },
        "cache": {
            "size": semantic_cache.size,
        },
    }
