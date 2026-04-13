# Architecture Design

## Overview

Smart RAG Agent is built around a **LangGraph state machine** that routes user queries through different retrieval strategies based on query analysis.

## Core Design: Adaptive Retrieval

The key insight is that not all questions need the same retrieval approach:

1. **General knowledge** → Direct LLM answer (no retrieval overhead)
2. **Specific documentation question** → Single retrieval with evaluation
3. **Complex multi-topic question** → Decompose into sub-queries, retrieve for each

### State Machine

```
AgentState:
  - question: str                     # Original user question
  - query_analysis: QueryAnalysis     # Route decision + reason
  - sub_queries: list[str]            # Decomposed sub-queries
  - current_query: str                # Current/rewritten query
  - retrieved_docs: list[Document]    # Retrieved documents (merged)
  - retrieval_attempts: int           # Retry counter
  - answer: str                       # Final answer
  - trace: list[TraceStep]            # Decision trace
```

### Routing Logic

The `analyze_query` node uses LLM with structured output to classify the query:

```python
class QueryAnalysis(BaseModel):
    action: "retrieve" | "direct_answer" | "decompose"
    reason: str
    rewritten_query: str       # Optimized for semantic search
    sub_queries: list[str]     # For decompose action
```

### Retrieval Pipeline

When retrieval is needed:

```
FAISS (semantic, top-20) ─┐
                           ├─→ RRF Fusion ─→ Reranker ─→ top-5
BM25 (keyword, top-20) ───┘
```

- **FAISS**: Captures semantic similarity
- **BM25**: Captures keyword matches that semantic search might miss
- **RRF (Reciprocal Rank Fusion)**: Merges two ranked lists fairly
- **Cross-Encoder Reranker**: Precise relevance scoring on merged candidates

### Self-Evaluation Loop

After retrieval, the agent evaluates whether the results are sufficient:

- **Sufficient** → Generate answer
- **Insufficient** → Rewrite query with different keywords, retry (max 3 times)

This prevents low-quality answers when the initial retrieval misses the mark.

## LLM Provider Abstraction

All providers use OpenAI-compatible APIs, unified through `ChatOpenAI`:

```python
def get_llm(config):
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,  # Different per provider
    )
```

Switching providers requires only changing `config.yaml`.

## Document Ingestion

Two ingestion paths:

1. **URL mode**: Crawler → HTML → Markdown → Chunking → FAISS
2. **Local mode**: File loader → Chunking → FAISS

The crawler respects same-domain boundaries, deduplicates URLs, and extracts main content by targeting `<main>`, `<article>`, or content divs.
