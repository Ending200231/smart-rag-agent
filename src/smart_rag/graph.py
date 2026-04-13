"""LangGraph agent state machine: adaptive retrieval decision graph."""

import operator
import time
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from smart_rag.config import RetrieverConfig
from smart_rag.planner import QueryAction, QueryAnalysis, analyze_query
from smart_rag.retriever import SmartRetriever, format_docs


# --- State Definition ---

@dataclass
class TraceStep:
    """A single step in the agent's decision trace."""
    node: str
    action: str
    detail: str
    duration_ms: float = 0.0


def _merge_docs(existing: list[Document], new: list[Document]) -> list[Document]:
    """Merge document lists, deduplicating by page_content."""
    seen = {doc.page_content for doc in existing}
    merged = list(existing)
    for doc in new:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged


class AgentState(TypedDict):
    question: str
    query_analysis: QueryAnalysis | None
    sub_queries: list[str]
    current_query: str  # Used for retry with rewritten query
    retrieved_docs: Annotated[list[Document], _merge_docs]
    retrieval_attempts: int  # Track retry count
    answer: str
    trace: Annotated[list[TraceStep], operator.add]


# --- Structured Output for Retrieval Evaluation ---

class RetrievalEvaluation(BaseModel):
    """Evaluation of whether retrieved documents are sufficient."""
    is_sufficient: bool = Field(description="Whether the retrieved documents contain enough information to answer the question")
    reason: str = Field(description="Brief explanation")
    rewritten_query: str = Field(
        default="",
        description="If not sufficient, a rewritten query to try. Empty if sufficient.",
    )


# --- Prompts ---

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个技术问答助手。这个问题不需要查阅文档，请直接根据你的知识回答。回答要准确、简洁。"),
    ("human", "{question}"),
])

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的技术文档问答助手。根据提供的参考文档回答用户的问题。

规则：
1. 综合所有参考文档中的信息回答，不要编造内容
2. 如果参考文档中没有相关信息，明确告知用户
3. 回答要准确、简洁、有条理
4. 在回答末尾标注引用的文档来源"""),
    ("human", """参考文档：
{context}

问题：{question}"""),
])

EVALUATE_RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval quality evaluator. Given a question and the retrieved documents, determine if the documents contain enough relevant information to provide a useful answer.

IMPORTANT: Be lenient in your evaluation. Mark as sufficient if:
- The documents contain information directly related to the topic, even if not perfectly comprehensive
- There are relevant code snippets, explanations, or references about the topic
- The documents provide enough context to give a helpful (even if partial) answer

Only mark as insufficient if:
- The documents are completely unrelated to the question
- The documents only mention the topic in passing without any useful detail

If insufficient, provide a rewritten query using different keywords."""),
    ("human", """Question: {question}

Retrieved documents:
{context}

Are these documents sufficient to answer the question?"""),
])


# --- Graph Builder ---

MAX_RETRIEVAL_ATTEMPTS = 3


def build_graph(
    llm: ChatOpenAI,
    vectorstore: FAISS,
    retriever_config: RetrieverConfig | None = None,
) -> StateGraph:
    """Build the adaptive retrieval agent graph with evaluation loop."""
    config = retriever_config or RetrieverConfig()
    retriever = SmartRetriever(vectorstore, config)

    def analyze_query_node(state: AgentState) -> dict:
        start = time.time()
        analysis = analyze_query(state["question"], llm)
        duration = (time.time() - start) * 1000

        trace_step = TraceStep(
            node="analyze_query",
            action=analysis.action.value,
            detail=analysis.reason,
            duration_ms=round(duration, 1),
        )

        result: dict = {
            "query_analysis": analysis,
            "trace": [trace_step],
            "retrieval_attempts": 0,
        }
        if analysis.action == QueryAction.DECOMPOSE:
            result["sub_queries"] = analysis.sub_queries
        if analysis.action == QueryAction.RETRIEVE:
            result["current_query"] = analysis.rewritten_query or state["question"]

        return result

    def retrieve_node(state: AgentState) -> dict:
        start = time.time()
        query = state.get("current_query") or state["question"]

        docs = retriever.retrieve(query)
        duration = (time.time() - start) * 1000
        attempts = state.get("retrieval_attempts", 0) + 1

        mode = "ensemble" if config.use_bm25 else "vector"
        if config.use_rerank:
            mode += "+rerank"

        trace_step = TraceStep(
            node="retrieve",
            action="search",
            detail=f"query='{query}', mode={mode}, found {len(docs)} docs (attempt {attempts})",
            duration_ms=round(duration, 1),
        )
        return {
            "retrieved_docs": docs,
            "retrieval_attempts": attempts,
            "trace": [trace_step],
        }

    def decompose_retrieve_node(state: AgentState) -> dict:
        start = time.time()
        sub_queries = state.get("sub_queries", [])
        all_docs: list[Document] = []
        details = []

        for sq in sub_queries:
            docs = retriever.retrieve(sq)
            all_docs.extend(docs)
            details.append(f"'{sq}' → {len(docs)} docs")

        duration = (time.time() - start) * 1000
        trace_step = TraceStep(
            node="decompose_retrieve",
            action="multi_search",
            detail="; ".join(details),
            duration_ms=round(duration, 1),
        )
        return {"retrieved_docs": all_docs, "trace": [trace_step]}

    def evaluate_retrieval_node(state: AgentState) -> dict:
        start = time.time()
        docs = state["retrieved_docs"]
        context = format_docs(docs)

        structured_llm = llm.with_structured_output(RetrievalEvaluation, method="function_calling")
        chain = EVALUATE_RETRIEVAL_PROMPT | structured_llm
        evaluation = chain.invoke({
            "question": state["question"],
            "context": context,
        })

        duration = (time.time() - start) * 1000
        trace_step = TraceStep(
            node="evaluate_retrieval",
            action="sufficient" if evaluation.is_sufficient else "insufficient",
            detail=evaluation.reason,
            duration_ms=round(duration, 1),
        )

        result: dict = {"trace": [trace_step]}
        if not evaluation.is_sufficient and evaluation.rewritten_query:
            result["current_query"] = evaluation.rewritten_query

        return result

    def generate_node(state: AgentState) -> dict:
        start = time.time()
        docs = state["retrieved_docs"]
        context = format_docs(docs)

        chain = SYNTHESIZE_PROMPT | llm
        response = chain.invoke({"context": context, "question": state["question"]})
        duration = (time.time() - start) * 1000

        trace_step = TraceStep(
            node="generate",
            action="answer",
            detail=f"based on {len(docs)} docs",
            duration_ms=round(duration, 1),
        )
        return {"answer": response.content, "trace": [trace_step]}

    def direct_answer_node(state: AgentState) -> dict:
        start = time.time()
        chain = DIRECT_ANSWER_PROMPT | llm
        response = chain.invoke({"question": state["question"]})
        duration = (time.time() - start) * 1000

        trace_step = TraceStep(
            node="direct_answer",
            action="answer",
            detail="answered without retrieval",
            duration_ms=round(duration, 1),
        )
        return {"answer": response.content, "trace": [trace_step]}

    def route_after_analysis(state: AgentState) -> str:
        action = state["query_analysis"].action
        if action == QueryAction.RETRIEVE:
            return "retrieve"
        elif action == QueryAction.DECOMPOSE:
            return "decompose_retrieve"
        else:
            return "direct_answer"

    def route_after_evaluation(state: AgentState) -> str:
        """Route based on evaluation: generate if sufficient, retry if not."""
        trace = state.get("trace", [])
        last_eval = None
        for step in reversed(trace):
            if step.node == "evaluate_retrieval":
                last_eval = step
                break

        if last_eval and last_eval.action == "insufficient":
            attempts = state.get("retrieval_attempts", 0)
            if attempts < MAX_RETRIEVAL_ATTEMPTS:
                return "retry"
            # Max attempts reached, generate with what we have
        return "generate"

    # --- Build Graph ---
    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decompose_retrieve", decompose_retrieve_node)
    graph.add_node("evaluate_retrieval", evaluate_retrieval_node)
    graph.add_node("generate", generate_node)
    graph.add_node("direct_answer", direct_answer_node)

    graph.set_entry_point("analyze_query")

    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "decompose_retrieve": "decompose_retrieve",
            "direct_answer": "direct_answer",
        },
    )

    # retrieve → evaluate → (generate | retry retrieve)
    graph.add_edge("retrieve", "evaluate_retrieval")
    graph.add_conditional_edges(
        "evaluate_retrieval",
        route_after_evaluation,
        {
            "generate": "generate",
            "retry": "retrieve",
        },
    )

    # decompose goes straight to generate (already multi-search)
    graph.add_edge("decompose_retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("direct_answer", END)

    return graph
