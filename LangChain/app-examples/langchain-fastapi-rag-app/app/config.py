from pydantic_settings import BaseSettings
from typing import Dict, Any


class Settings(BaseSettings):
    # 应用配置
    app_name: str = "RAG Application"
    app_version: str = "1.0.0"

    # 向量数据库配置
    chroma_persist_dir: str = "./chroma_db"
    chroma_collections: Dict[str, Dict[str, Any]] = {
        "product": {
            "source_dir": "data/products",
            "embedding": "openai",
            "chunk_size": 1000
        },
        "technical": {
            "source_dir": "data/technical",
            "embedding": "sentence-transformers",
            "chunk_size": 1500
        }
    }

    # OpenAI 配置
    openai_api_key: str
    openai_model: str = "gpt-4-turbo"

    # 检索配置
    default_top_k: int = 5
    similarity_threshold: float = 0.75

    # 文档处理配置
    watch_delay: int = 5  # 文件监控延迟(秒)
    max_workers: int = 4  # 并行处理线程数

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()