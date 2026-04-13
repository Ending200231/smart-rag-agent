"""Evaluation script: Baseline RAG vs Adaptive Agent comparison."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from smart_rag.agent import Agent
from smart_rag.config import Config
from smart_rag.indexer import load_index
from smart_rag.llm import get_llm
from smart_rag.retriever import format_docs

console = Console()


# --- LLM-as-Judge Scoring ---

class AnswerScore(BaseModel):
    """Scoring result from LLM judge."""
    faithfulness: float = Field(description="0-1 score: is the answer grounded in the provided documents? 1=fully grounded, 0=hallucinated")
    relevancy: float = Field(description="0-1 score: does the answer address the question? 1=perfectly relevant, 0=irrelevant")
    completeness: float = Field(description="0-1 score: does the answer cover the key aspects? 1=complete, 0=missing key info")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an impartial judge evaluating a Q&A system's response. Score the answer on three dimensions, each 0.0 to 1.0.

1. **faithfulness**: Is the answer grounded in the provided source documents? Penalize hallucinated claims not in the sources. If no sources were used (direct answer), score based on factual accuracy.
2. **relevancy**: Does the answer actually address the question asked?
3. **completeness**: Does the answer cover the key aspects of the question? A partial but accurate answer should get 0.5-0.7.

Be fair and consistent. A good but not perfect answer should score 0.7-0.85."""),
    ("human", """Question: {question}

Answer: {answer}

Source documents used:
{sources}

Score this answer."""),
])


@dataclass
class EvalResult:
    """Result for a single question."""
    question_id: int
    question: str
    category: str
    expected_route: str
    actual_route: str = ""
    answer: str = ""
    faithfulness: float = 0.0
    relevancy: float = 0.0
    completeness: float = 0.0
    route_correct: bool = False
    num_docs: int = 0
    latency_ms: float = 0.0
    num_retrieval_attempts: int = 0


@dataclass
class EvalSummary:
    """Aggregated evaluation results."""
    mode: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        return sum(r.faithfulness for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_relevancy(self) -> float:
        return sum(r.relevancy for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_completeness(self) -> float:
        return sum(r.completeness for r in self.results) / len(self.results) if self.results else 0

    @property
    def route_accuracy(self) -> float:
        routable = [r for r in self.results if r.expected_route]
        return sum(1 for r in routable if r.route_correct) / len(routable) if routable else 0

    @property
    def avg_latency(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_docs(self) -> float:
        return sum(r.num_docs for r in self.results) / len(self.results) if self.results else 0


def load_dataset(path: str = "eval/dataset.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def run_baseline(questions: list[dict], config: Config) -> EvalSummary:
    """Run baseline RAG (always retrieve, no routing, no rerank)."""
    console.print("[bold blue]Running Baseline RAG...[/]")

    vectorstore = load_index(config)
    llm = get_llm(config.llm)
    judge_llm = llm.with_structured_output(AnswerScore, method="function_calling")

    from smart_rag.retriever import RAG_PROMPT

    summary = EvalSummary(mode="Baseline RAG")

    for i, q in enumerate(questions, 1):
        console.print(f"  [{i}/{len(questions)}] {q['question'][:50]}...")

        start = time.time()
        docs = vectorstore.similarity_search(q["question"], k=config.retriever.top_k)
        context = format_docs(docs)
        chain = RAG_PROMPT | llm
        response = chain.invoke({"context": context, "question": q["question"]})
        latency = (time.time() - start) * 1000

        # Judge
        sources_text = format_docs(docs) if docs else "(no sources)"
        score = judge_llm.invoke(
            JUDGE_PROMPT.format_messages(
                question=q["question"],
                answer=response.content,
                sources=sources_text,
            )
        )

        result = EvalResult(
            question_id=q["id"],
            question=q["question"],
            category=q["category"],
            expected_route=q["expected_route"],
            actual_route="retrieve",  # Baseline always retrieves
            answer=response.content,
            faithfulness=score.faithfulness,
            relevancy=score.relevancy,
            completeness=score.completeness,
            route_correct=(q["expected_route"] == "retrieve"),
            num_docs=len(docs),
            latency_ms=round(latency, 1),
            num_retrieval_attempts=1,
        )
        summary.results.append(result)

    return summary


def run_agent(questions: list[dict], config: Config) -> EvalSummary:
    """Run full adaptive agent."""
    console.print("[bold blue]Running Adaptive Agent...[/]")

    agent = Agent(config=config)
    llm = get_llm(config.llm)
    judge_llm = llm.with_structured_output(AnswerScore, method="function_calling")

    summary = EvalSummary(mode="Adaptive Agent")

    for i, q in enumerate(questions, 1):
        console.print(f"  [{i}/{len(questions)}] {q['question'][:50]}...")

        start = time.time()
        result = agent.ask(q["question"])
        latency = (time.time() - start) * 1000

        # Determine actual route from trace
        actual_route = "unknown"
        num_attempts = 0
        if result.trace:
            for step in result.trace:
                if step.node == "analyze_query":
                    actual_route = step.action
                if step.node == "retrieve":
                    num_attempts += 1

        # Judge
        sources_text = format_docs(result.sources) if result.sources else "(no sources - direct answer)"
        score = judge_llm.invoke(
            JUDGE_PROMPT.format_messages(
                question=q["question"],
                answer=result.text,
                sources=sources_text,
            )
        )

        eval_result = EvalResult(
            question_id=q["id"],
            question=q["question"],
            category=q["category"],
            expected_route=q["expected_route"],
            actual_route=actual_route,
            answer=result.text,
            faithfulness=score.faithfulness,
            relevancy=score.relevancy,
            completeness=score.completeness,
            route_correct=(q["expected_route"] == actual_route),
            num_docs=len(result.sources),
            latency_ms=round(latency, 1),
            num_retrieval_attempts=num_attempts,
        )
        summary.results.append(eval_result)

    return summary


def print_comparison(baseline: EvalSummary, agent: EvalSummary):
    """Print side-by-side comparison table."""
    table = Table(title="Evaluation Results: Baseline vs Adaptive Agent", show_lines=True)
    table.add_column("Metric", style="bold", width=25)
    table.add_column("Baseline RAG", justify="center", width=15)
    table.add_column("Adaptive Agent", justify="center", width=15)
    table.add_column("Diff", justify="center", width=10)

    metrics = [
        ("Faithfulness", baseline.avg_faithfulness, agent.avg_faithfulness),
        ("Relevancy", baseline.avg_relevancy, agent.avg_relevancy),
        ("Completeness", baseline.avg_completeness, agent.avg_completeness),
        ("Route Accuracy", None, agent.route_accuracy),
        ("Avg Latency (ms)", baseline.avg_latency, agent.avg_latency),
        ("Avg Docs Retrieved", baseline.avg_docs, agent.avg_docs),
    ]

    for name, base_val, agent_val in metrics:
        if base_val is None:
            table.add_row(name, "N/A", f"{agent_val:.2f}", "")
        else:
            diff = agent_val - base_val
            diff_str = f"[green]+{diff:.2f}[/]" if diff > 0 else f"[red]{diff:.2f}[/]"
            if name == "Avg Latency (ms)":
                diff_str = f"[green]{diff:.0f}[/]" if diff < 0 else f"[red]+{diff:.0f}[/]"
                table.add_row(name, f"{base_val:.0f}", f"{agent_val:.0f}", diff_str)
            else:
                table.add_row(name, f"{base_val:.2f}", f"{agent_val:.2f}", diff_str)

    console.print()
    console.print(table)

    # Category breakdown
    categories = sorted(set(r.category for r in agent.results))
    cat_table = Table(title="Agent Route Accuracy by Category", show_lines=True)
    cat_table.add_column("Category", style="bold", width=15)
    cat_table.add_column("Total", justify="center", width=8)
    cat_table.add_column("Correct", justify="center", width=8)
    cat_table.add_column("Accuracy", justify="center", width=10)

    for cat in categories:
        cat_results = [r for r in agent.results if r.category == cat]
        correct = sum(1 for r in cat_results if r.route_correct)
        total = len(cat_results)
        acc = correct / total if total > 0 else 0
        cat_table.add_row(cat, str(total), str(correct), f"{acc:.0%}")

    console.print()
    console.print(cat_table)


def save_results(baseline: EvalSummary, agent: EvalSummary, output_path: str = "eval/results.json"):
    """Save detailed results to JSON."""
    def summary_to_dict(s: EvalSummary) -> dict:
        return {
            "mode": s.mode,
            "metrics": {
                "avg_faithfulness": round(s.avg_faithfulness, 3),
                "avg_relevancy": round(s.avg_relevancy, 3),
                "avg_completeness": round(s.avg_completeness, 3),
                "route_accuracy": round(s.route_accuracy, 3),
                "avg_latency_ms": round(s.avg_latency, 1),
                "avg_docs": round(s.avg_docs, 1),
            },
            "details": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "category": r.category,
                    "expected_route": r.expected_route,
                    "actual_route": r.actual_route,
                    "route_correct": r.route_correct,
                    "faithfulness": r.faithfulness,
                    "relevancy": r.relevancy,
                    "completeness": r.completeness,
                    "num_docs": r.num_docs,
                    "latency_ms": r.latency_ms,
                }
                for r in s.results
            ],
        }

    output = {
        "baseline": summary_to_dict(baseline),
        "agent": summary_to_dict(agent),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    console.print(f"\n[dim]Detailed results saved to {output_path}[/]")


def run_evaluation(config: Config, dataset_path: str = "eval/dataset.json"):
    """Run the full evaluation pipeline."""
    questions = load_dataset(dataset_path)
    console.print(f"[bold]Loaded {len(questions)} evaluation questions[/]")

    baseline = run_baseline(questions, config)
    agent_results = run_agent(questions, config)

    print_comparison(baseline, agent_results)
    save_results(baseline, agent_results)
