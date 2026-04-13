"""Smart RAG Agent - Adaptive retrieval-augmented generation with autonomous decision-making."""

__version__ = "0.1.0"

from smart_rag.agent import Agent, AgentResponse
from smart_rag.graph import TraceStep

__all__ = ["Agent", "AgentResponse", "TraceStep"]
