from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
from app.utils.logger import logger

def get_embedding_model(model_name: str):
    """获取嵌入模型实例"""
    if model_name == "openai":
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model="text-embedding-3-small"
        )
    elif model_name == "sentence-transformers":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        logger.error(f"未知的嵌入模型: {model_name}")
        raise ValueError(f"不支持的嵌入模型: {model_name}")