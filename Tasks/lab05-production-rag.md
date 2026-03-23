# Lab 05: Production RAG System with Evaluation

## Objective

Build and deploy a production-ready Retrieval-Augmented Generation (RAG) system with comprehensive evaluation, prompt caching, rate limiting, and observability. This lab brings together retrieval (Lab 04), tool use (Lab 02), and production patterns into a complete system.


---

## Learning Goals

- Build an end-to-end RAG pipeline (ingest, retrieve, generate, cite)
- Implement RAG evaluation metrics (Precision@K, MRR, faithfulness)
- Use LLM-as-judge for generation quality evaluation
- Add production patterns: rate limiting, prompt caching, error handling
- Deploy with metrics and monitoring endpoints

---

## What You Will Build

A production RAG system that:

1. Indexes a codebase or documentation set
2. Answers questions with source citations
3. Evaluates retrieval quality (Precision@K, MRR)
4. Evaluates generation quality (LLM-as-judge for faithfulness)
5. Implements rate limiting and prompt caching
6. Exposes a metrics endpoint for monitoring

---

## Architecture

```
                         Indexing
                         =======
POST /index ──> [Chunk] ──> [Embed] ──> Vector Store (Pinecone / SQLite)
                  │                          │
                  └──> [Tokenize] ──> BM25 Index


                         Query
                         =====
POST /query ──> [Rate Limit] ──> [Semantic Cache Check]
                                        │
                                  cache miss
                                        │
                                        v
                                 [Hybrid Retrieve]
                                        │
                                        v
                                 [Build Prompt w/ Context]
                                        │
                                        v
                                 [Gemini Generate]
                                        │
                                        v
                                 [Extract Citations]
                                        │
                                        v
                                 [Log Metrics] ──> Response


                         Evaluation
                         ==========
POST /evaluate ──> [Load eval dataset] ──> [Run queries]
                                                │
                                    ┌───────────┼───────────┐
                                    v           v           v
                              [Precision@K]  [MRR]   [LLM-as-Judge]
                                    │           │           │
                                    └───────────┴───────────┘
                                                │
                                                v
                                          Eval Report
```

---

## Prerequisites

- Google AI API key (`GOOGLE_API_KEY`) -- free tier available at [aistudio.google.com](https://aistudio.google.com)
- Pinecone API key (free tier, or use SQLite with vectors from Lab 04)
- `sentence-transformers` (Python) or Ollama (TypeScript) for local embeddings -- free
- Python 3.11+ or Node.js 20+
- Redis (optional, for rate limiting -- can use in-memory fallback)

---

## Step 1: RAG Pipeline Core

Build the retrieval and generation pipeline. This is the central module that takes a user query plus retrieved context chunks and produces a cited answer via Gemini.

### What to build

Create a module (`rag.py` or `src/rag.ts`) that contains:

1. **A system prompt** that instructs the model to answer only from provided context, cite sources using `[Source N]` notation, and admit when information is insufficient.

2. **A response data structure** (`RAGResponse`) with fields: `answer` (string), `sources` (list of source objects with index, text preview, and metadata), `latency_ms` (float), and `cache_hit` (boolean).

3. **A metrics tracker** (`RAGMetrics`) that accumulates: `total_queries`, `total_latency_ms`, `cache_hits`, `errors`. It should expose computed properties for `avg_latency_ms` and `cache_hit_rate`. Keep this as a module-level singleton.

4. **A `build_context` function** that takes a list of retrieved chunks and formats them into a numbered context block. Each chunk should be labeled `[Source N]` with its metadata title.

5. **An async `generate_answer` function** that:
   - Accepts a query string, a list of context chunks, and a cache flag
   - Builds the context string from chunks
   - Constructs a user message containing the context and the question
   - Calls Gemini (`gemini-2.0-flash`) with the system prompt
   - Measures latency in milliseconds
   - Updates the global metrics tracker
   - Returns a `RAGResponse`
   - Increments the error counter on failure before re-raising

### Hints

- **Python**: Use `google.generativeai` with `genai.GenerativeModel`. The `system_instruction` parameter sets the system prompt at model creation time. Use `time.time()` for latency measurement.
- **TypeScript**: Use `@google/generative-ai` with `GoogleGenerativeAI`. Use `Date.now()` for timing.
- Initialize the model once at module level, not per request.
- Keep the source preview in each response truncated to ~200 characters.

### How to verify

- Import your module and call `generate_answer` with a dummy query and a list of fake chunks. You should get back a `RAGResponse` with a non-empty `answer` and correct `sources` length.
- After several calls, `metrics.total_queries` should match the call count and `avg_latency_ms` should be a reasonable positive number.
- If you pass an empty context, the model should respond indicating insufficient information (per the system prompt rules).

---

## Step 2: Rate Limiter

Protect the API from abuse and control costs.

### What to build

Create a rate limiter module (`rate_limiter.py` or `src/rateLimiter.ts`) with:

1. **A configuration object** with three limits: `requests_per_minute` (default 20), `requests_per_hour` (default 200), `tokens_per_minute` (default 100,000).

2. **An `InMemoryRateLimiter` class** implementing a sliding window algorithm:
   - Maintain per-client timestamp lists for minute and hour windows
   - A `check_rate_limit(client_id)` method that:
     - Cleans expired entries from each window (older than 60s or 3600s)
     - Checks if the count exceeds the configured limit
     - Raises an HTTP 429 error (or returns an error object) if exceeded
     - Records the current timestamp if allowed
   - A `record_tokens(client_id, token_count)` method to track token usage per minute
   - A `get_token_usage(client_id)` method that returns the current minute's total

3. **A module-level singleton** instance of the rate limiter.

### Hints

- The sliding window approach stores timestamps of each request and filters out entries older than the window size.
- **Python**: Use `collections.defaultdict(list)` for the window storage. Raise `fastapi.HTTPException` with status 429.
- **TypeScript**: Use a `Map<string, number[]>` for windows. Return `{ ok: false, error: "..." }` instead of throwing.
- Token counts need a tuple/pair of `(timestamp, count)` since each request may consume a different number of tokens.

### How to verify

- Create a limiter with `requests_per_minute=3`. Call `check_rate_limit("test")` four times in quick succession. The fourth call should raise/return a 429 error.
- Wait 60+ seconds (or manipulate timestamps in a test) and confirm the limiter resets.
- Call `record_tokens("test", 500)` twice, then verify `get_token_usage("test")` returns 1000.

---

## Step 3: Semantic Cache

Cache responses for semantically similar queries to avoid redundant API calls.

### What to build

Create a semantic cache module (`semantic_cache.py` or `src/semanticCache.ts`) with:

1. **A `CacheEntry` data structure** holding: `query` (string), `embedding` (float array), `response` (dict/object), `created_at` (timestamp), and `ttl_seconds` (default 3600).

2. **A `SemanticCache` class** with:
   - Constructor parameters: `similarity_threshold` (default 0.95) and `max_entries` (default 1000)
   - A **cosine similarity** static method. The formula is:

     ```
     cosine_similarity(a, b) = (a . b) / (||a|| * ||b||)
     ```

     where `a . b` is the dot product and `||x||` is the Euclidean norm (square root of sum of squares).

   - A `get(query_embedding)` method that:
     - Removes expired entries (older than their TTL)
     - Scans all entries, computes cosine similarity against the query embedding
     - Returns the response of the best match if similarity >= threshold, otherwise `None`/`null`

   - A `put(query, query_embedding, response, ttl_seconds)` method that:
     - Evicts the oldest 25% of entries if at capacity
     - Appends a new `CacheEntry`

   - A `size` property returning the current entry count

3. **A module-level singleton** instance.

### Hints

- The cosine similarity computation is straightforward: iterate once through both vectors computing dot product and both norms simultaneously.
- Handle the zero-norm edge case (return 0.0).
- For TTL expiration, compare `current_time - created_at` against `ttl_seconds`.
- The 25% eviction strategy prevents constant eviction when at capacity.

### How to verify

- Create a cache, store a response with a known embedding, then query with the exact same embedding. You should get the cached response back.
- Query with a very different embedding (e.g., all zeros vs. all ones). You should get `None`/`null`.
- Store entries, wait for TTL expiration (use a short TTL like 1 second in tests), and confirm they are cleaned up.
- Fill the cache to `max_entries` and add one more. Verify the size dropped by ~25% before the new entry was added.

---

## Step 4: RAG Evaluation Suite

Build evaluation metrics for both retrieval quality and generation quality.

### What to build

Create an evaluation module (`evaluation.py` or `src/evaluation.ts`) with:

1. **An `EvalResult` data structure** with fields: `precision_at_k`, `mrr`, `faithfulness`, `relevance` (all floats), and `details` (list of per-query result dicts).

2. **`precision_at_k(retrieved_ids, relevant_ids, k)`** -- Computes what fraction of the top-K retrieved documents are relevant.

   ```
   Precision@K = |{relevant docs in top-K}| / K
   ```

   - `retrieved_ids`: ordered list of document IDs returned by the retriever
   - `relevant_ids`: set of ground-truth relevant document IDs
   - `k`: how many top results to consider
   - Return 0.0 if top-K is empty

3. **`mean_reciprocal_rank(retrieved_ids, relevant_ids)`** -- Returns 1/rank of the first relevant document found.

   ```
   MRR = 1 / rank_of_first_relevant_document
   ```

   - Iterate through `retrieved_ids` in order; the rank is 1-indexed
   - Return 0.0 if no relevant document is found

4. **`llm_judge_faithfulness(query, answer, context)`** -- Uses Gemini as a judge to score whether the answer is faithful to the provided context on a 0.0-1.0 scale. The prompt should:
   - Present the context, question, and answer
   - Define the scoring rubric (1.0 = fully supported, 0.5 = partially, 0.0 = contradicts/fabricates)
   - Request a JSON response with `score` and `explanation` fields
   - Parse the JSON; return score 0.0 with an error explanation if parsing fails

5. **`llm_judge_relevance(query, answer)`** -- Same pattern as faithfulness but scores whether the answer addresses the question (no context needed in prompt).

6. **`run_evaluation(eval_dataset, retrieval_fn, generation_fn, k)`** -- Orchestrator that:
   - Iterates over each item in the eval dataset (each has `query` and `relevant_doc_ids`)
   - Calls `retrieval_fn(query)` to get chunks, extracts their IDs
   - Computes Precision@K and MRR per query
   - Calls `generation_fn(query, chunks)` to get an answer
   - Runs both LLM judges on the answer
   - Collects all scores and returns an `EvalResult` with averaged metrics

### Hints

- Use a separate Gemini model instance for the judge (same `gemini-2.0-flash` model is fine).
- The LLM judge prompts should ask for "ONLY a JSON object" to make parsing reliable.
- Handle JSON parsing failures gracefully -- return a 0 score with the raw text as explanation.
- `retrieval_fn` and `generation_fn` are async callbacks, allowing the evaluation suite to be decoupled from specific implementations.

### How to verify

- Test `precision_at_k(["a","b","c"], {"a","c"}, 3)` -- should return `2/3 = 0.6667`.
- Test `precision_at_k(["a","b","c"], {"d"}, 3)` -- should return `0.0`.
- Test `mean_reciprocal_rank(["a","b","c"], {"b"})` -- should return `0.5`.
- Test `mean_reciprocal_rank(["a","b","c"], {"d"})` -- should return `0.0`.
- Call the faithfulness judge with a context and an answer that clearly contradicts it. Expect a low score.
- Run the full evaluation with a small 2-3 item dataset and verify the output contains all four averaged metrics plus per-query details.

---

## Step 5: Build the Complete API

Assemble all components into a production-ready API with the following endpoints.

### What to build

Create a main application file (`main.py` with FastAPI, or `src/main.ts` with Hono) that wires together the RAG core, rate limiter, semantic cache, and evaluation suite.

**Endpoints to implement:**

1. **`POST /query`** -- Main RAG query endpoint. Request body: `query`, `limit` (default 5), `use_cache` (default true), `client_id` (default "default"). The handler should execute this pipeline in order:
   - Check rate limit for the client
   - Embed the query
   - If caching is enabled, check the semantic cache; on hit, increment metrics and return early
   - Retrieve chunks using hybrid search (reuse your Lab 04 retrieval or stub it)
   - Generate an answer via your RAG module
   - Store the response in the semantic cache
   - Record token usage for rate limiting
   - Return the answer, sources, latency, and cache_hit flag

2. **`POST /index`** -- Accepts `doc_id`, `title`, `text`. Plug in your Lab 04 indexing pipeline or return a stub acknowledgement.

3. **`POST /evaluate`** -- Accepts a `dataset` (list of `{query, relevant_doc_ids}`) and `k`. Wires up `retrieval_fn` and `generation_fn` callbacks, calls `run_evaluation`, and returns the full evaluation report.

4. **`GET /metrics`** -- Returns the RAG metrics (from your metrics singleton) and cache stats (cache size).

5. **`GET /health`** -- Returns `{"status": "ok"}`.

**Helper functions to implement:**

- `embed_query(query)` -- Generates an embedding vector for the query.
  - Python: use `sentence-transformers` with `all-MiniLM-L6-v2`
  - TypeScript: use Ollama's `/api/embeddings` endpoint with `all-minilm`
- `retrieve(query, limit)` -- Calls your hybrid search from Lab 04 (or returns placeholder data while developing).

### Hints

- **Python**: Use `FastAPI`, `pydantic.BaseModel` for request/response models, `uvicorn` to serve.
- **TypeScript**: Use `hono` with `@hono/node-server` for serving.
- The `/query` endpoint is the most complex -- it chains rate limiting, caching, retrieval, generation, and metrics in sequence. Pay attention to the order of operations.
- Load the embedding model once at module level, not per request (especially for sentence-transformers which is heavy).

### Install dependencies

**Python:**
```bash
uv add fastapi uvicorn google-generativeai sentence-transformers pinecone
uv run uvicorn main:app --reload --port 8000
```

**TypeScript:**
```bash
npm install hono @hono/node-server @google/generative-ai @pinecone-database/pinecone
npm install -D typescript tsx @types/node
```

### How to verify

Start the server and run these requests:

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the chunking strategy work?", "client_id": "user1"}'
# Expected: JSON with answer, sources, latency_ms, cache_hit fields

# Repeat the same query -- cache_hit should be true on the second call

# Metrics
curl http://localhost:8000/metrics
# Expected: JSON like:
# {
#   "rag": { "total_queries": 2, "avg_latency_ms": ..., "cache_hit_rate": 0.5, "errors": 0 },
#   "cache": { "size": 1 }
# }
```

---

## Step 6: Test and Deploy

### Test the evaluation endpoint

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": [
      {
        "query": "What is hybrid search?",
        "relevant_doc_ids": ["doc1_abc123"]
      },
      {
        "query": "How does BM25 scoring work?",
        "relevant_doc_ids": ["doc1_def456"]
      }
    ],
    "k": 5
  }'
```

The response should contain `precision_at_k`, `mrr`, `faithfulness`, `relevance` averages and a `details` array with per-query breakdowns.

### Deploy

```bash
railway login
railway init
railway variables set GOOGLE_API_KEY=your-google-api-key
railway variables set PINECONE_API_KEY=...
railway up
```

### How to verify

- All five endpoints (`/health`, `/query`, `/index`, `/evaluate`, `/metrics`) respond correctly.
- Rate limiting triggers a 429 after exceeding the configured threshold.
- The second identical query returns `cache_hit: true` with lower latency.
- The evaluation endpoint returns numeric scores for all four metrics.
- The deployed version is accessible at your Railway/Render URL.

---

## Deliverables

- [ ] End-to-end RAG pipeline (index, retrieve, generate with citations)
- [ ] Evaluation suite with Precision@K, MRR, and LLM-as-judge (faithfulness + relevance)
- [ ] Rate limiting middleware (per-minute and per-hour)
- [ ] Semantic caching for repeated/similar queries
- [ ] Metrics endpoint for monitoring
- [ ] Deployed to Railway or Render

---

## Extension Challenges

1. **Add Self-RAG** -- Before retrieving, use Gemini to decide whether retrieval is needed. For simple factual questions the model already knows, skip retrieval entirely. This reduces latency and cost. Prompt the model with the query and ask it to respond with YES or NO to whether external documents are needed.

2. **Tune the semantic cache threshold** -- Experiment with the `similarity_threshold` parameter: lower values (0.90) increase cache hits but risk returning stale answers; higher values (0.98) are safer but cache less. Run your evaluation suite at different thresholds and compare metrics.

3. **Add A/B testing for chunking strategies** -- Index the same documents with two different chunking strategies (e.g., 300-char vs. 800-char chunks). Route queries randomly to each strategy, track evaluation metrics per strategy, and compare.

4. **Build automated evaluation on a schedule** -- Create a cron job or scheduled task that runs the evaluation suite nightly against a golden dataset and logs the results. Alert if metrics drop below a threshold.

---

## Key Production Patterns Summary

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| Rate Limiting | Prevent abuse, control costs | Sliding window counter per client |
| Semantic Caching | Avoid redundant API calls for similar queries | Embedding similarity on query vectors |
| LLM-as-Judge | Evaluate generation quality without human labels | Gemini scores faithfulness and relevance |
| Metrics Endpoint | Monitor system health | Track latency, cache rates, errors |
| Error Handling | Graceful degradation | Try/catch with fallback responses |

---

## References

- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [RAG Evaluation Best Practices](https://docs.ragas.io/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)
