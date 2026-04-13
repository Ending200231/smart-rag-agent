"""Document loading, chunking, and FAISS index building."""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smart_rag.config import Config
from smart_rag.embedding import get_embedding


def load_documents(docs_path: str) -> list:
    """Load documents from a directory of Markdown/text files."""
    path = Path(docs_path)
    if not path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_path}")

    # Load .md and .mdx files
    loaders = []
    for ext in ["**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst"]:
        loader = DirectoryLoader(
            str(path),
            glob=ext,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        loaders.append(loader)

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No documents found in {docs_path}")

    print(f"Loaded {len(docs)} documents from {docs_path}")
    return docs


def chunk_documents(docs: list, chunk_size: int = 512, chunk_overlap: int = 50) -> list:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def build_index(docs_path: str, config: Config | None = None) -> FAISS:
    """Load documents, chunk them, and build a FAISS index."""
    if config is None:
        config = Config.default()

    docs = load_documents(docs_path)
    chunks = chunk_documents(
        docs,
        chunk_size=config.retriever.chunk_size,
        chunk_overlap=config.retriever.chunk_overlap,
    )

    embedding = get_embedding(config.embedding)
    vectorstore = FAISS.from_documents(chunks, embedding)

    # Persist index to disk
    index_dir = Path(config.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Index saved to {index_dir}")

    return vectorstore


def load_index(config: Config | None = None) -> FAISS:
    """Load a previously built FAISS index from disk."""
    if config is None:
        config = Config.default()

    index_dir = Path(config.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index not found at {index_dir}. Run 'smart-rag index --docs <path>' first."
        )

    embedding = get_embedding(config.embedding)
    vectorstore = FAISS.load_local(
        str(index_dir), embedding, allow_dangerous_deserialization=True
    )
    print(f"Index loaded from {index_dir}")
    return vectorstore
