"""Retriever with FAISS + BM25 multi-path recall and optional Reranking."""

from dataclasses import dataclass

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from smart_rag.config import Config, RetrieverConfig
from smart_rag.llm import get_llm


@dataclass
class RetrievalResult:
    """Result from retrieval + generation."""
    answer: str
    sources: list[Document]


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的技术文档问答助手。根据提供的参考文档回答用户的问题。

规则：
1. 只根据参考文档中的信息回答，不要编造内容
2. 如果参考文档中没有相关信息，明确告知用户
3. 回答要准确、简洁、有条理
4. 在回答末尾标注引用的文档来源"""),
    ("human", """参考文档：
{context}

问题：{question}"""),
])


class SmartRetriever:
    """Multi-path retriever with optional BM25 and Reranking."""

    def __init__(self, vectorstore: FAISS, config: RetrieverConfig | None = None):
        self.vectorstore = vectorstore
        self.config = config or RetrieverConfig()
        self._bm25 = None
        self._bm25_docs = None
        self._reranker = None

        if self.config.use_bm25:
            self._init_bm25()
        if self.config.use_rerank:
            self._init_reranker()

    def _init_bm25(self):
        """Build BM25 index from all documents in the vectorstore."""
        # Extract all docs from FAISS
        docstore = self.vectorstore.docstore
        index_to_id = self.vectorstore.index_to_docstore_id
        self._bm25_docs = []
        for i in range(len(index_to_id)):
            doc_id = index_to_id[i]
            doc = docstore.search(doc_id)
            if doc:
                self._bm25_docs.append(doc)

        # Tokenize for BM25 (simple whitespace + punctuation split)
        tokenized = [self._tokenize(doc.page_content) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    def _init_reranker(self):
        """Load the cross-encoder reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                self.config.rerank_model,
                device=None,  # Auto-detect GPU/CPU
            )
        except ImportError:
            print("Warning: sentence-transformers not installed.")
            self.config.use_rerank = False

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization for BM25."""
        import re
        return re.findall(r'\w+', text.lower())

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents using configured strategy."""
        if self.config.use_bm25 and self._bm25:
            return self._ensemble_retrieve(query)
        else:
            return self._vector_retrieve(query)

    def _vector_retrieve(self, query: str) -> list[Document]:
        """Pure vector similarity search."""
        k = self.config.top_k_initial if self.config.use_rerank else self.config.top_k
        docs = self.vectorstore.similarity_search(query, k=k)

        if self.config.use_rerank and self._reranker:
            docs = self._rerank(query, docs)

        return docs

    def _ensemble_retrieve(self, query: str) -> list[Document]:
        """Combine vector search + BM25, then optionally rerank."""
        k_initial = self.config.top_k_initial if self.config.use_rerank else self.config.top_k

        # Vector search
        vector_docs = self.vectorstore.similarity_search(query, k=k_initial)

        # BM25 search
        tokenized_query = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:k_initial]
        bm25_docs = [self._bm25_docs[i] for i in top_indices if bm25_scores[i] > 0]

        # Merge and deduplicate (RRF - Reciprocal Rank Fusion)
        merged = self._reciprocal_rank_fusion(vector_docs, bm25_docs, k=k_initial)

        if self.config.use_rerank and self._reranker:
            merged = self._rerank(query, merged)

        return merged

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Rerank documents using cross-encoder."""
        if not docs:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._reranker.predict(pairs)

        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.config.top_k]]

    @staticmethod
    def _reciprocal_rank_fusion(
        list_a: list[Document],
        list_b: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """Merge two ranked lists using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: dict[str, tuple[float, Document]] = {}

        for rank, doc in enumerate(list_a):
            key = doc.page_content[:200]
            score = 1.0 / (rank + k)
            if key in rrf_scores:
                rrf_scores[key] = (rrf_scores[key][0] + score, doc)
            else:
                rrf_scores[key] = (score, doc)

        for rank, doc in enumerate(list_b):
            key = doc.page_content[:200]
            score = 1.0 / (rank + k)
            if key in rrf_scores:
                rrf_scores[key] = (rrf_scores[key][0] + score, doc)
            else:
                rrf_scores[key] = (score, doc)

        sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_results]


def retrieve_and_generate(
    question: str,
    vectorstore: FAISS,
    config: Config | None = None,
) -> RetrievalResult:
    """Retrieve relevant documents and generate an answer (Phase 1 baseline)."""
    if config is None:
        config = Config.default()

    docs = vectorstore.similarity_search(question, k=config.retriever.top_k)
    context = format_docs(docs)

    llm = get_llm(config.llm)
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    return RetrievalResult(answer=response.content, sources=docs)


def format_docs(docs: list[Document]) -> str:
    """Format documents into a context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[文档{i}] (来源: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# Keep old name for backward compatibility
_format_docs = format_docs
