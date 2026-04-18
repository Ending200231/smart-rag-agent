"""Configuration management for Smart RAG Agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048

    # Provider presets: (default_model, default_base_url)
    PROVIDER_PRESETS: dict = field(default_factory=lambda: {
        "deepseek": ("deepseek-chat", "https://api.deepseek.com/v1"),
        "zhipu": ("glm-4-flash", "https://open.bigmodel.cn/api/paas/v4"),
        "qwen": ("qwen-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "openai": ("gpt-4o-mini", "https://api.openai.com/v1"),
    }, repr=False)

    def __post_init__(self):
        preset = self.PROVIDER_PRESETS.get(self.provider, {})
        if isinstance(preset, tuple):
            default_model, default_base_url = preset
        else:
            default_model, default_base_url = "", ""

        if not self.model or self.model == "deepseek-chat":
            self.model = default_model or self.model
        if not self.base_url:
            self.base_url = default_base_url

        # API key from env if not set
        env_key_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "zhipu": "ZHIPUAI_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        if not self.api_key:
            env_var = env_key_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
            self.api_key = os.environ.get(env_var, "")


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-base-zh-v1.5"
    device: str = "cuda"


@dataclass
class RetrieverConfig:
    top_k: int = 5
    top_k_initial: int = 20  # Before rerank
    chunk_size: int = 512
    chunk_overlap: int = 50
    rerank_model: str = "BAAI/bge-reranker-base"
    use_rerank: bool = False  # Phase 3 enables this
    use_bm25: bool = False    # Phase 3 enables this
    use_hyde: bool = False    # HyDE: hypothetical document embeddings


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    index_dir: str = "./index"
    verbose: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        llm_data = data.get("llm", {})
        embedding_data = data.get("embedding", {})
        retriever_data = data.get("retriever", {})

        return cls(
            llm=LLMConfig(**llm_data),
            embedding=EmbeddingConfig(**embedding_data),
            retriever=RetrieverConfig(**retriever_data),
            index_dir=data.get("index_dir", "./index"),
            verbose=data.get("verbose", False),
        )

    @classmethod
    def default(cls) -> "Config":
        return cls()
