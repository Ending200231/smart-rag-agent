# Smart RAG Agent

[English](README.md) | [中文](README_zh.md)

An adaptive RAG (Retrieval-Augmented Generation) agent that **autonomously decides retrieval strategies** instead of blindly retrieving for every query.

Give it a documentation site URL or a local directory — it crawls, indexes, and answers your questions intelligently.

## Why This Project?

Traditional RAG has three pain points:

| Problem | Traditional RAG | Smart RAG Agent |
|---------|----------------|-----------------|
| **Noise interference** | Retrieves for every query, even simple ones | Routes simple questions directly to LLM |
| **Incomplete retrieval** | Single retrieval may miss relevant info | Decomposes complex questions into sub-queries |
| **No quality control** | Blindly uses whatever is retrieved | Evaluates retrieval quality, retries with rewritten query if insufficient |

## Architecture

```
                    ┌─────────────────┐
                    │  User Question  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Analyze Query  │ ← LLM judges question type
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼────┐ ┌───────▼───────┐
     │ Direct Answer  │ │Retrieve│ │   Decompose   │
     │ (no retrieval) │ │        │ │ (sub-queries) │
     └────────┬───────┘ └───┬────┘ └───────┬───────┘
              │             │              │
              │    ┌────────▼────────┐     │
              │    │ Evaluate Result │     │
              │    └────────┬────────┘     │
              │        ┌────┴────┐         │
              │   sufficient  insufficient │
              │        │      ↓ rewrite    │
              │        │    [retry]        │
              │    ┌───▼────┐              │
              │    │Generate│◄─────────────┘
              │    └───┬────┘
              │        │
              └────────┼────────┐
                       │
              ┌────────▼────────┐
              │  Agent Response │
              │ text + sources  │
              │    + trace      │
              └─────────────────┘
```

**Key Design Decisions:**
- **LangGraph** for state machine orchestration (not hand-written loops)
- **Adaptive routing**: 3 strategies — direct answer, single retrieval, decompose & multi-retrieve
- **Self-evaluation**: Agent judges if retrieved docs are sufficient, retries with rewritten query (max 3 attempts)
- **Multi-path recall**: FAISS (semantic) + BM25 (keyword) + Reranker (cross-encoder)

## Quick Start

### 1. Install

```bash
git clone https://github.com/Ending200231/smart-rag-agent.git
cd smart-rag-agent
python -m venv venv
# Linux/Mac: source venv/bin/activate
# Windows: venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your API key:
# DEEPSEEK_API_KEY=sk-xxx

cp config.example.yaml config.yaml
```

### 3. Index Documentation

```bash
# From a URL (auto-crawl)
smart-rag index --url https://fastapi.tiangolo.com --config config.yaml

# Or from local files
smart-rag index --docs ./my-docs --config config.yaml
```

### 4. Chat

```bash
smart-rag chat --config config.yaml --verbose
```

### 5. Use as Library

```python
from smart_rag import Agent

agent = Agent(index_dir="./index")
response = agent.ask("How to use dependency injection in FastAPI?")

print(response.text)       # Answer
print(response.sources)    # Source documents
print(response.trace)      # Decision trace
```

## Features

- **Auto-crawl documentation sites** — just give a URL
- **Local docs support** — Markdown, MDX, TXT, RST files
- **Adaptive retrieval** — Agent decides when and how to retrieve
- **Multi-path recall** — FAISS + BM25 + cross-encoder reranking
- **Self-evaluation loop** — retries with rewritten queries when results are insufficient
- **Decision trace** — full transparency into the agent's reasoning
- **Multi-LLM support** — DeepSeek, Zhipu, Qwen, OpenAI (all via unified interface)
- **Local embedding & reranking** — BGE models on GPU, no API calls needed

## Configuration

Edit `config.yaml`:

```yaml
llm:
  provider: deepseek          # deepseek / zhipu / qwen / openai
  temperature: 0.0

embedding:
  model_name: BAAI/bge-base-zh-v1.5
  device: cuda                # cuda / cpu

retriever:
  top_k: 5
  chunk_size: 512
  use_bm25: true              # Enable BM25 keyword search
  use_rerank: true             # Enable cross-encoder reranking
```

## Evaluation

Run the built-in evaluation comparing Baseline RAG vs Adaptive Agent:

```bash
smart-rag eval --config config.yaml
```

The evaluation covers 40 questions across 4 categories:
- **Simple** (15): direct documentation questions
- **Complex** (10): multi-topic questions requiring decomposition
- **General** (7): common knowledge, should skip retrieval
- **Out-of-scope** (5): topics not in the docs, should acknowledge gaps

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Agent Orchestration | LangGraph | State machine with conditional routing, built for agent workflows |
| LLM Interface | LangChain + ChatOpenAI | Unified API for multiple providers |
| Embedding | BGE-base-zh-v1.5 (local GPU) | Free, bilingual, good quality |
| Vector Store | FAISS | Lightweight, no external service needed |
| Keyword Search | BM25 (rank_bm25) | Complements semantic search |
| Reranker | BGE-reranker-base (local GPU) | Cross-encoder precision on top of recall |
| Crawler | requests + BeautifulSoup + html2text | Simple, reliable, no headless browser needed |

## Project Structure

```
smart-rag-agent/
├── src/smart_rag/
│   ├── agent.py          # Public API: Agent class
│   ├── graph.py          # LangGraph state machine
│   ├── planner.py        # Query analysis & routing
│   ├── retriever.py      # Multi-path retrieval + rerank
│   ├── crawler.py        # Documentation site crawler
│   ├── indexer.py        # Document chunking & FAISS indexing
│   ├── embedding.py      # BGE embedding wrapper
│   ├── llm.py            # Unified LLM interface
│   ├── config.py         # Configuration management
│   └── cli.py            # CLI commands
├── eval/                 # Evaluation dataset & scripts
├── examples/             # Usage examples
└── docs/                 # Architecture documentation
```

## License

MIT
