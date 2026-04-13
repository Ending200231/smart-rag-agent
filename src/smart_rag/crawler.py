"""Documentation site crawler: URL → local Markdown files."""

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress

console = Console()


class DocCrawler:
    """Crawl a documentation site and convert pages to Markdown."""

    def __init__(
        self,
        base_url: str,
        output_dir: str = "./crawled_docs",
        max_pages: int = 500,
        max_depth: int = 5,
        delay: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.domain = urlparse(self.base_url).netloc
        self.output_dir = Path(output_dir) / self.domain
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay

        self.visited: set[str] = set()
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        self.h2t.body_width = 0  # No line wrapping

    def crawl(self) -> Path:
        """Crawl the site and save as Markdown files. Returns output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold blue]Crawling:[/] {self.base_url}")
        console.print(f"[dim]Output: {self.output_dir} | Max pages: {self.max_pages}[/]")

        with Progress() as progress:
            task = progress.add_task("[cyan]Crawling pages...", total=None)
            self._crawl_page(self.base_url, depth=0, progress=progress, task=task)
            progress.update(task, total=len(self.visited), completed=len(self.visited))

        console.print(f"[bold green]Done![/] Crawled {len(self.visited)} pages → {self.output_dir}")
        return self.output_dir

    def _crawl_page(self, url: str, depth: int, progress: Progress, task) -> None:
        """Recursively crawl a single page."""
        normalized = self._normalize_url(url)
        if normalized in self.visited:
            return
        if len(self.visited) >= self.max_pages:
            return
        if depth > self.max_depth:
            return

        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "SmartRAGAgent/0.1"})
            resp.raise_for_status()
        except requests.RequestException as e:
            console.print(f"[dim red]  Skip {url}: {e}[/]")
            return

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            return

        self.visited.add(normalized)
        progress.update(task, completed=len(self.visited))

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract main content (try common doc site selectors)
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|doc|markdown", re.I))
            or soup.find("body")
        )

        if main_content:
            # Remove nav, footer, sidebar
            for tag in main_content.find_all(["nav", "footer", "aside", "header"]):
                tag.decompose()

            markdown = self.h2t.handle(str(main_content))
            title = soup.title.string if soup.title else normalized

            # Save to file
            self._save_page(normalized, title, markdown, url)

        # Find links and recurse
        if depth < self.max_depth:
            time.sleep(self.delay)
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if self._is_same_site(next_url):
                    self._crawl_page(next_url, depth + 1, progress, task)

    def _save_page(self, normalized: str, title: str, content: str, source_url: str) -> None:
        """Save a page as a Markdown file."""
        # Create filename from URL path
        path = urlparse(normalized).path.strip("/")
        if not path:
            path = "index"
        # Replace / with _ for flat file structure
        filename = path.replace("/", "_").replace(".", "_") + ".md"
        # Sanitize filename
        filename = re.sub(r'[<>:"|?*]', "_", filename)

        filepath = self.output_dir / filename

        # Add metadata header
        full_content = f"---\ntitle: {title}\nsource: {source_url}\n---\n\n{content}"
        filepath.write_text(full_content, encoding="utf-8")

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Remove fragment and trailing slash
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def _is_same_site(self, url: str) -> bool:
        """Check if URL belongs to the same documentation site."""
        parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return False
        # Skip non-doc resources
        skip_exts = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".zip", ".pdf"}
        if any(parsed.path.lower().endswith(ext) for ext in skip_exts):
            return False
        # Skip fragments-only links
        if not parsed.path or parsed.path == "#":
            return False
        return True
