# Smart RAG Agent

[English](README.md) | [中文](README_zh.md)

一个**自适应 RAG（检索增强生成）Agent**，能够自主决策检索策略，而非对每个查询无脑检索。

给它一个文档站 URL 或本地目录，它会自动爬取、索引，然后智能回答你的问题。

## 为什么做这个项目？

传统 RAG 有三个痛点：

| 问题 | 传统 RAG | Smart RAG Agent |
|------|---------|-----------------|
| **噪音干扰** | 每次都检索，简单问题也不例外 | 简单问题直接用 LLM 回答，跳过检索 |
| **检索不全** | 单次检索可能遗漏相关信息 | 复杂问题分解为子查询，多次检索 |
| **无质量控制** | 检索到什么就用什么 | 评估检索质量，不够好则改写 query 重试 |

## 架构

```
                       ┌─────────────────┐
                       │  User Question  │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
                       │  Analyze Query  │  LLM judges question type
                       └────────┬────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
   ┌────────▼────────┐   ┌──────▼──────┐  ┌─────────▼─────────┐
   │  Direct Answer  │   │  Retrieve   │  │    Decompose      │
   │  (skip retrieval│   │  FAISS+BM25 │  │                   │
   │   LLM answers   │   │  +Rerank    │  │   (sub-queries)   │
   │   directly)     │   │             │  │                   │
   └────────┬────────┘   └──────┬──────┘  └─────────┬─────────┘
            │                   │                   │
            │          ┌───────▼────────┐           │
            │          │Evaluate Result │           │
            │          └───────┬────────┘           │
            │             ┌────┴─────┐              │
            │        sufficient  insufficient       │
            │             │       ↓ rewrite         │
            │             │     [retry, max 3]      │
            │             │                         │
            │          ┌──▼──────────┐              │
            │          │  Generate   │◄─────────────┘
            │          │ with context│
            │          └──────┬──────┘
            │                 │
            └───────┬─────────┘
                    │
           ┌────────▼────────┐
           │  Agent Response │
           │ text + sources  │
           │    + trace      │
           └─────────────────┘
```

> Analyze Query = 分析问题 | Direct Answer = 直接回答（无需检索，LLM 直接生成） | Retrieve = 检索 | Decompose = 分解子问题 | Evaluate Result = 评估检索质量 | Generate = 基于检索结果生成回答
>
> **说明：** Direct Answer 和 Generate 是**两个独立的生成节点**。Direct Answer 不带检索上下文直接生成（通用知识）；Generate 基于检索到的文档内容生成（文档相关问题）。

**核心设计决策：**
- **LangGraph** 编排状态机（非手写循环）
- **自适应路由**：3 种策略 — 直接回答、单次检索、分解多次检索
- **自评估机制**：Agent 判断检索结果是否充分，不充分则改写 query 重试（最多 3 次）
- **多路召回**：FAISS（语义）+ BM25（关键词）+ Reranker（交叉编码器精排）

## 快速开始

### 1. 安装

```bash
git clone https://github.com/Ending200231/smart-rag-agent.git
cd smart-rag-agent
python -m venv venv
# Linux/Mac: source venv/bin/activate
# Windows: venv\Scripts\Activate.ps1
pip install -e .
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env 添加你的 API Key：
# DEEPSEEK_API_KEY=sk-xxx

cp config.example.yaml config.yaml
```

### 3. 索引文档

```bash
# 从 URL 自动爬取
smart-rag index --url https://fastapi.tiangolo.com --config config.yaml

# 或从本地文件
smart-rag index --docs ./my-docs --config config.yaml
```

### 4. 对话

```bash
smart-rag chat --config config.yaml --verbose
```

### 5. 作为库使用

```python
from smart_rag import Agent

agent = Agent(index_dir="./index")
response = agent.ask("FastAPI 中如何使用依赖注入？")

print(response.text)       # 回答
print(response.sources)    # 来源文档
print(response.trace)      # 决策轨迹
```

## 功能特性

- **自动爬取文档站** — 只需提供 URL
- **本地文档支持** — Markdown、MDX、TXT、RST 文件
- **自适应检索** — Agent 自主决策何时检索、如何检索
- **多路召回** — FAISS + BM25 + 交叉编码器重排序
- **自评估闭环** — 检索不充分时自动改写 query 重试
- **决策轨迹** — 完全透明的推理过程
- **多模型支持** — DeepSeek、智谱、通义、OpenAI（统一接口切换）
- **本地 Embedding & Rerank** — BGE 模型 GPU 推理，无需 API 调用

## 配置说明

编辑 `config.yaml`：

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
  use_bm25: true              # 启用 BM25 关键词检索
  use_rerank: true            # 启用交叉编码器重排序
```

## 评估

运行内置评测，对比 Baseline RAG 与 Adaptive Agent：

```bash
smart-rag eval --config config.yaml
```

评测覆盖 40 个问题，分 4 类：
- **简单问题**（15 个）：直接的文档问题
- **复杂问题**（10 个）：需要分解的多主题问题
- **通用问题**（7 个）：常识问题，应跳过检索
- **超范围问题**（5 个）：文档未覆盖的话题，应坦诚告知

### 评测结果

| 指标 | Baseline RAG | Adaptive Agent | 变化 |
|------|-------------|---------------|------|
| Faithfulness（忠实度） | 0.96 | 0.88 | -0.08 |
| Relevancy（相关性） | 0.51 | 0.89 | **+0.38** |
| Completeness（完整性） | 0.26 | 0.67 | **+0.41** |
| 路由准确率 | N/A | 0.70 | — |

> 回答相关性和完整性大幅提升。Faithfulness 略降是因为 Agent 对通用问题直接回答（无文档锚定），这是合理的 trade-off。

## 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| Agent 编排 | LangGraph | 状态机 + 条件路由，专为 Agent 设计 |
| LLM 接口 | LangChain + ChatOpenAI | 统一多厂商 API |
| Embedding | BGE-base-zh-v1.5（本地 GPU） | 免费、中英双语、质量好 |
| 向量存储 | FAISS | 轻量、无需外部服务 |
| 关键词检索 | BM25 (rank_bm25) | 补充语义检索的不足 |
| 重排序 | BGE-reranker-base（本地 GPU） | 在召回基础上精确排序 |
| 爬虫 | requests + BeautifulSoup + html2text | 简单可靠，无需无头浏览器 |

## 项目结构

```
smart-rag-agent/
├── src/smart_rag/
│   ├── agent.py          # 公开 API：Agent 类
│   ├── graph.py          # LangGraph 状态机
│   ├── planner.py        # 查询分析与路由
│   ├── retriever.py      # 多路召回 + 重排序
│   ├── crawler.py        # 文档站爬虫
│   ├── indexer.py        # 文档分块 + FAISS 索引
│   ├── embedding.py      # BGE Embedding 封装
│   ├── llm.py            # 统一 LLM 接口
│   ├── config.py         # 配置管理
│   └── cli.py            # CLI 命令
├── eval/                 # 评测数据集与脚本
├── examples/             # 使用示例
└── docs/                 # 架构设计文档
```

## License

MIT
