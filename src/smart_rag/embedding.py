"""BGE Embedding model wrapper with query instruction support."""

from langchain_huggingface import HuggingFaceEmbeddings

from smart_rag.config import EmbeddingConfig

# BGE models need this prefix on queries (not documents) for optimal retrieval
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BgeEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings with BGE query instruction prefix."""

    query_instruction: str = BGE_QUERY_INSTRUCTION

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self.query_instruction + text)


def get_embedding(config: EmbeddingConfig | None = None) -> BgeEmbeddings:
    """Create a BGE embedding model instance."""
    if config is None:
        config = EmbeddingConfig()

    return BgeEmbeddings(
        model_name=config.model_name,
        model_kwargs={"device": config.device},
        encode_kwargs={"normalize_embeddings": True},
    )
