"""
RAG Evaluation Suite — Step 4

WHY evaluate a RAG system?
A RAG system can fail in TWO different places:
  1. RETRIEVAL failure — the right documents weren't found
  2. GENERATION failure — the right docs were found but the LLM gave a bad answer

So we need TWO types of metrics:

  Retrieval metrics (did we find the right docs?):
    - Precision@K: of the top K docs retrieved, how many were actually relevant?
    - MRR (Mean Reciprocal Rank): how high up did the first relevant doc appear?

  Generation metrics (did the LLM answer well?):
    - Faithfulness: does the answer stick to the provided context? (no hallucination)
    - Relevance: does the answer actually address the question?

For generation metrics, we use "LLM-as-Judge" — we ask an LLM to score
the answers. This is cheaper and faster than human evaluation, and research
shows it correlates well with human judgement.


RETRIEVAL METRICS — Visual Examples:

Precision@K (K=5):
  Retrieved: [doc_A, doc_B, doc_C, doc_D, doc_E]
  Relevant:  {doc_A, doc_C, doc_F}
                ✓      ✗      ✓      ✗      ✗
  Precision@5 = 2 relevant found / 5 total = 0.40

  Think of it as: "Of the 5 documents I showed the model, what % were useful?"


MRR (Mean Reciprocal Rank):
  Retrieved: [doc_X, doc_Y, doc_A, doc_Z]
  Relevant:  {doc_A}
               ✗      ✗      ✓ ← first relevant at position 3
  MRR = 1/3 = 0.333

  Think of it as: "How quickly did I find a relevant document?"
  - Position 1 → MRR = 1.0 (perfect!)
  - Position 2 → MRR = 0.5
  - Position 5 → MRR = 0.2
  - Not found  → MRR = 0.0
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from openai import OpenAI

# Lazy init — same pattern as rag.py. The judge client is created on first
# use, AFTER main.py has loaded the .env file with the API key.
_judge_client = None


def _get_judge_client() -> OpenAI:
    """Initialize OpenAI client for the judge on first call."""
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _judge_client


def _judge_call(prompt: str) -> str:
    """Helper: send a prompt to gpt-4o-mini and return the text response."""
    response = _get_judge_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    """
    Aggregated evaluation results across all queries in the dataset.

    The top-level floats are AVERAGES across all queries.
    The details list has per-query breakdowns for debugging.
    """
    precision_at_k: float
    mrr: float
    faithfulness: float
    relevance: float
    details: list[dict]


# --------------------------------------------------------------------------- #
# Retrieval Metrics
# --------------------------------------------------------------------------- #
def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Precision@K — what fraction of the top-K retrieved docs are relevant?

    Formula:
        Precision@K = |{relevant docs in top-K}| / K

    Example:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant  = {"a", "c"}
        k = 3
        top_3 = ["a", "b", "c"]
        relevant in top_3 = {"a", "c"} → 2
        Precision@3 = 2/3 = 0.667

    Why this matters:
        If Precision@5 = 0.2, it means 4 out of 5 chunks shown to the LLM
        were irrelevant noise. The model has to work harder to find the
        answer, and might get confused by the irrelevant chunks.
    """
    if k == 0:
        return 0.0

    # Take only the top K results
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    # Count how many of the top K are in the relevant set
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_count / k


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    MRR — 1 / (position of first relevant document).

    Iterates through retrieved docs in order (1-indexed).
    Returns 0.0 if no relevant document is found at all.

    Example:
        retrieved = ["x", "y", "a", "z"]
        relevant  = {"a", "b"}
        → "a" found at position 3
        → MRR = 1/3 = 0.333

    Why this matters:
        MRR tells you how quickly the retriever surfaces a relevant doc.
        If MRR is consistently low, users are getting answers based on
        chunks found deep in the results (or not at all).
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):  # 1-indexed!
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------- #
# LLM-as-Judge — Generation Metrics
# --------------------------------------------------------------------------- #
async def llm_judge_faithfulness(
    query: str,
    answer: str,
    context: str,
) -> dict:
    """
    Uses OpenAI as a judge to score: does the answer stay faithful to the context?

    "Faithful" means:
      - The answer only claims things that appear in the context
      - It doesn't hallucinate facts not present in the sources
      - It doesn't contradict information in the sources

    Scoring rubric:
      1.0 = fully supported by context
      0.5 = partially supported (some claims lack evidence)
      0.0 = contradicts context or fabricates information

    Why LLM-as-Judge?
      Human evaluation is gold standard but expensive and slow.
      Research shows that strong LLMs (GPT-4, etc.) correlate ~85-90%
      with human judgement on faithfulness scoring. Good enough for
      automated evaluation loops.
    """
    prompt = f"""You are an evaluation judge. Score whether the answer is faithful to the provided context.

Context:
{context}

Question: {query}

Answer: {answer}

Scoring rubric:
- 1.0: The answer is fully supported by the context. All claims can be traced to the sources.
- 0.5: The answer is partially supported. Some claims are supported, but others lack evidence in the context.
- 0.0: The answer contradicts the context or fabricates information not present in the sources.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"score": <float between 0 and 1>, "explanation": "<brief explanation>"}}"""

    try:
        raw = _judge_call(prompt)
        result = json.loads(raw)
        return {
            "score": float(result.get("score", 0.0)),
            "explanation": result.get("explanation", ""),
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return {
            "score": 0.0,
            "explanation": f"Failed to parse judge response: {raw[:200]}",
        }


async def llm_judge_relevance(
    query: str,
    answer: str,
) -> dict:
    """
    Uses OpenAI to score: does the answer actually address the question?

    Different from faithfulness:
      - Faithfulness = "is the answer grounded in the sources?"
      - Relevance = "does the answer address what was asked?"

    An answer can be faithful but irrelevant (correctly cites sources but
    answers a different question), or relevant but unfaithful (addresses
    the question but makes things up).

    We need BOTH metrics to catch different failure modes.
    """
    prompt = f"""You are an evaluation judge. Score whether the answer is relevant to the question.

Question: {query}

Answer: {answer}

Scoring rubric:
- 1.0: The answer directly and completely addresses the question.
- 0.5: The answer partially addresses the question or includes significant irrelevant information.
- 0.0: The answer does not address the question at all.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"score": <float between 0 and 1>, "explanation": "<brief explanation>"}}"""

    try:
        raw = _judge_call(prompt)
        result = json.loads(raw)
        return {
            "score": float(result.get("score", 0.0)),
            "explanation": result.get("explanation", ""),
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return {
            "score": 0.0,
            "explanation": f"Failed to parse judge response: {raw[:200]}",
        }


# --------------------------------------------------------------------------- #
# Evaluation Orchestrator
# --------------------------------------------------------------------------- #
async def run_evaluation(
    eval_dataset: list[dict],
    retrieval_fn: Callable[[str], Awaitable[list[dict]]],
    generation_fn: Callable[[str, list[dict]], Awaitable[str]],
    k: int = 5,
) -> EvalResult:
    """
    Runs the full evaluation pipeline over a dataset.

    For each item in the dataset:
      1. Call retrieval_fn(query) → get chunks
      2. Compute Precision@K and MRR (retrieval quality)
      3. Call generation_fn(query, chunks) → get answer
      4. Run both LLM judges (generation quality)
      5. Collect all scores

    Finally, average everything and return an EvalResult.

    Args:
        eval_dataset: list of {"query": str, "relevant_doc_ids": list[str]}
        retrieval_fn: async function that takes a query and returns chunks
                      (each chunk must have an "id" field)
        generation_fn: async function that takes (query, chunks) and returns answer text
        k: how many top results to evaluate for Precision@K
    """
    all_precision = []
    all_mrr = []
    all_faithfulness = []
    all_relevance = []
    details = []

    for item in eval_dataset:
        query = item["query"]
        relevant_ids = set(item["relevant_doc_ids"])

        # Step 1: Retrieve
        chunks = await retrieval_fn(query)
        retrieved_ids = [chunk.get("id", "") for chunk in chunks]

        # Step 2: Retrieval metrics
        p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
        mrr_score = mean_reciprocal_rank(retrieved_ids, relevant_ids)

        # Step 3: Generate answer
        answer = await generation_fn(query, chunks)

        # Step 4: Build context string for faithfulness judge
        from rag import build_context
        context = build_context(chunks)

        # Step 5: LLM judges
        faith_result = await llm_judge_faithfulness(query, answer, context)
        relev_result = await llm_judge_relevance(query, answer)

        # Collect scores
        all_precision.append(p_at_k)
        all_mrr.append(mrr_score)
        all_faithfulness.append(faith_result["score"])
        all_relevance.append(relev_result["score"])

        # Per-query detail for debugging
        details.append({
            "query": query,
            "precision_at_k": p_at_k,
            "mrr": mrr_score,
            "faithfulness": faith_result,
            "relevance": relev_result,
            "retrieved_ids": retrieved_ids,
            "answer_preview": answer[:200],
        })

    # Average all metrics across the dataset
    n = len(eval_dataset) or 1  # avoid division by zero
    return EvalResult(
        precision_at_k=sum(all_precision) / n,
        mrr=sum(all_mrr) / n,
        faithfulness=sum(all_faithfulness) / n,
        relevance=sum(all_relevance) / n,
        details=details,
    )
