"""CLI entry point for Smart RAG Agent."""

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from smart_rag.config import Config
from smart_rag.crawler import DocCrawler
from smart_rag.indexer import build_index

load_dotenv()
console = Console()


def _load_config(config_path: str | None) -> Config:
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return Config.default()


@click.group()
def main():
    """Smart RAG Agent - Adaptive retrieval-augmented generation.

    Give it a documentation URL or local directory, and chat with your docs.
    """
    pass


@main.command()
@click.option("--url", default=None, help="Documentation site URL to crawl")
@click.option("--docs", default=None, help="Local documents directory path")
@click.option("--config", "config_path", default=None, help="Path to config YAML")
@click.option("--max-pages", default=200, help="Max pages to crawl (URL mode)")
def index(url: str | None, docs: str | None, config_path: str | None, max_pages: int):
    """Build index from a documentation URL or local directory."""
    if not url and not docs:
        raise click.UsageError("Provide either --url or --docs")

    config = _load_config(config_path)

    if url:
        console.print(f"[bold blue]Step 1/2: Crawling[/] {url}")
        crawler = DocCrawler(base_url=url, max_pages=max_pages)
        docs_path = str(crawler.crawl())
    else:
        docs_path = docs
        console.print(f"[bold blue]Loading local docs from[/] {docs_path}")

    console.print("[bold blue]Step 2/2: Building index...[/]" if url else "[bold blue]Building index...[/]")
    build_index(docs_path, config)
    console.print("[bold green]Index built successfully![/]")


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML")
@click.option("--verbose", is_flag=True, help="Show source documents and trace")
def chat(config_path: str | None, verbose: bool):
    """Interactive chat with the RAG agent."""
    from smart_rag.agent import Agent

    config = _load_config(config_path)
    if verbose:
        config.verbose = True

    console.print(Panel(
        "[bold]Smart RAG Agent[/] (Adaptive Mode)\n"
        "Ask questions about your indexed documentation.\n"
        "Type [bold cyan]quit[/] to exit.",
        border_style="blue",
    ))

    with console.status("[bold yellow]Loading agent...[/]"):
        agent = Agent(config=config)

    while True:
        console.print()
        question = console.input("[bold cyan]> [/]").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/]")
            break

        with console.status("[bold yellow]Thinking...[/]"):
            result = agent.ask(question)

        # Display answer
        console.print()
        console.print(Markdown(result.text))

        # Display trace
        if config.verbose and result.trace:
            console.print()
            trace_table = Table(title="Decision Trace", show_lines=True)
            trace_table.add_column("Node", style="cyan", width=20)
            trace_table.add_column("Action", style="green", width=15)
            trace_table.add_column("Detail", style="dim")
            trace_table.add_column("Time(ms)", style="yellow", justify="right", width=10)
            for step in result.trace:
                trace_table.add_row(step.node, step.action, step.detail, str(step.duration_ms))
            console.print(trace_table)

        # Display sources
        if config.verbose and result.sources:
            console.print()
            console.print("[dim]--- Sources ---[/]")
            seen = set()
            for i, doc in enumerate(result.sources, 1):
                source = doc.metadata.get("source", "unknown")
                if source in seen:
                    continue
                seen.add(source)
                console.print(f"[dim]  [{i}] {source}[/]")


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML")
@click.option("--dataset", default="eval/dataset.json", help="Path to evaluation dataset")
def eval(config_path: str | None, dataset: str):
    """Run evaluation: Baseline RAG vs Adaptive Agent."""
    import sys
    from pathlib import Path

    # Add project root to sys.path so eval/ is importable
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from eval.run_eval import run_evaluation

    config = _load_config(config_path)
    run_evaluation(config, dataset)


if __name__ == "__main__":
    main()
