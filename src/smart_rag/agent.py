"""Agent main interface: the public API for Smart RAG Agent."""

from dataclasses import dataclass, field

from langchain_core.documents import Document

from smart_rag.config import Config
from smart_rag.graph import TraceStep, build_graph
from smart_rag.indexer import load_index
from smart_rag.llm import get_llm


@dataclass
class AgentResponse:
    """Structured response from the agent."""
    text: str
    sources: list[Document] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)


class Agent:
    """Smart RAG Agent with adaptive retrieval strategies.

    Usage:
        agent = Agent(index_dir="./index")
        response = agent.ask("How to use dependency injection in FastAPI?")
        print(response.text)
        print(response.sources)
        print(response.trace)
    """

    def __init__(self, index_dir: str | None = None, config: Config | None = None):
        self.config = config or Config.default()
        if index_dir:
            self.config.index_dir = index_dir

        self.llm = get_llm(self.config.llm)
        self.vectorstore = load_index(self.config)

        graph = build_graph(
            llm=self.llm,
            vectorstore=self.vectorstore,
            retriever_config=self.config.retriever,
        )
        self.app = graph.compile()

    def ask(self, question: str) -> AgentResponse:
        """Ask a question and get an adaptive response."""
        initial_state = {
            "question": question,
            "query_analysis": None,
            "sub_queries": [],
            "current_query": "",
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "answer": "",
            "trace": [],
        }

        result = self.app.invoke(initial_state)

        return AgentResponse(
            text=result["answer"],
            sources=result.get("retrieved_docs", []),
            trace=result.get("trace", []),
        )
