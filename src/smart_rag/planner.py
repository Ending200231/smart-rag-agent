"""Query analysis: routing, rewriting, and decomposition."""

from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class QueryAction(str, Enum):
    RETRIEVE = "retrieve"
    DIRECT_ANSWER = "direct_answer"
    DECOMPOSE = "decompose"


class QueryAnalysis(BaseModel):
    """Result of analyzing a user query."""
    action: QueryAction = Field(description="The action to take for this query")
    reason: str = Field(description="Brief explanation of why this action was chosen")
    rewritten_query: str = Field(
        default="",
        description="Rewritten query for better retrieval (only for retrieve action)",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-queries (only for decompose action)",
    )


ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query analyzer for a documentation Q&A system. Your job is to analyze the user's question and decide the best strategy.

The knowledge base contains technical documentation. Analyze the query and choose ONE action:

1. **retrieve**: The question is about specific topics covered in the documentation. Use this for most documentation-related questions. Also provide a rewritten_query that is optimized for semantic search (clear, specific, in the same language as the docs).

2. **direct_answer**: The question is general knowledge NOT related to the documentation (e.g., "what is Python?", "explain HTTP status codes"). The LLM can answer directly without retrieval.

3. **decompose**: The question is complex and requires information from multiple topics. Break it down into 2-4 simpler sub-queries that can each be answered with a single retrieval. Each sub-query should be self-contained and specific.

Respond in the structured format requested."""),
    ("human", "{question}"),
])


def analyze_query(question: str, llm: ChatOpenAI) -> QueryAnalysis:
    """Analyze a query to determine the retrieval strategy."""
    structured_llm = llm.with_structured_output(QueryAnalysis, method="function_calling")
    chain = ANALYZE_PROMPT | structured_llm
    return chain.invoke({"question": question})
