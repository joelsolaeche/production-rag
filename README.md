# Production RAG System

A production-ready Retrieval-Augmented Generation system with evaluation, semantic caching, rate limiting, and observability.

**Live:** https://production-rag.onrender.com

## Architecture

```
POST /query ──> [Rate Limit] ──> [Semantic Cache Check]
                                        │
                                  cache miss
                                        │
                                        v
                                 [Embed Query (OpenAI)]
                                        │
                                        v
                                 [Hybrid Retrieve]
                                        │
                                        v
                                 [Generate Answer (GPT-4o-mini)]
                                        │
                                        v
                                 [Extract Citations]
                                        │
                                        v
                                 [Log Metrics] ──> Response
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/query` | Main RAG pipeline — returns cited answer |
| `POST` | `/index` | Add documents to the knowledge base |
| `POST` | `/evaluate` | Run evaluation suite (Precision@K, MRR, LLM-as-judge) |
| `GET` | `/metrics` | System health: latency, cache hit rate, errors |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Install & Run

```bash
# Install dependencies
uv sync

# Set your API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# Start the server
uv run uvicorn main:app --reload --port 8000
```

### Test the endpoints

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does hybrid search work?", "client_id": "user1"}'

# Same query again — cache_hit should be true
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does hybrid search work?", "client_id": "user1"}'

# Check metrics
curl http://localhost:8000/metrics

# Run evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": [
      {"query": "What is hybrid search?", "relevant_doc_ids": ["doc1_chunk1"]},
      {"query": "How does BM25 scoring work?", "relevant_doc_ids": ["doc1_chunk2"]}
    ],
    "k": 5
  }'
```

## Project Structure

```
production-rag/
├── main.py              # FastAPI app — 5 endpoints, wires all modules
├── rag.py               # Core RAG pipeline (system prompt, generate, metrics)
├── rate_limiter.py      # Sliding window rate limiter (per-minute, per-hour, tokens)
├── semantic_cache.py    # Cosine similarity cache on query embeddings
├── evaluation.py        # Precision@K, MRR, LLM-as-judge (faithfulness + relevance)
├── requirements.txt     # Dependencies for deployment
├── render.yaml          # Render deployment config
└── .env                 # API keys (not committed)
```

## Production Patterns

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| **Rate Limiting** | Prevent abuse, control costs | Sliding window counter per client (minute + hour) |
| **Semantic Caching** | Skip redundant LLM calls for similar queries | Cosine similarity on OpenAI embeddings (threshold: 0.95) |
| **LLM-as-Judge** | Evaluate generation quality without human labels | GPT-4o-mini scores faithfulness and relevance |
| **Metrics Endpoint** | Monitor system health | Tracks latency, cache rates, error counts |
| **Citation Extraction** | Traceable answers | `[Source N]` notation mapped to retrieved chunks |

## Evaluation Metrics

**Retrieval quality:**
- **Precision@K** — fraction of top-K retrieved docs that are relevant
- **MRR** — reciprocal rank of first relevant document

**Generation quality (LLM-as-Judge):**
- **Faithfulness** — is the answer grounded in the provided sources?
- **Relevance** — does the answer address the question asked?

## Tech Stack

- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** OpenAI text-embedding-3-small
- **Framework:** FastAPI
- **Deployment:** Render
