import os
import time
from langchain_community.vectorstores import Chroma
from app.config import settings
from app.core.embeddings import get_embedding_model
from app.services.document_loader import DocumentLoader
from app.utils.logger import logger
from app.core.versioning import needs_update


class VectorStoreService:
    def __init__(self):
        self.stores = {}
        self.needs_update = {}
        self.last_processed = {}

    def mark_for_update(self, doc_type: str):
        """标记文档类型需要更新"""
        self.needs_update[doc_type] = True
        logger.info(f"标记向量库需要更新: {doc_type}")

    def init_store(self, doc_type: str, force: bool = False):
        """初始化或加载向量存储"""
        start_time = time.time()
        config = settings.chroma_collections[doc_type]
        persist_dir = os.path.join(settings.chroma_persist_dir, doc_type)
        embedding = get_embedding_model(config["embedding"])

        # 检查是否需要更新
        needs_rebuild = force or self.needs_update.get(doc_type, False)

        if not needs_rebuild:
            # 检查持久化数据是否存在
            if os.path.exists(persist_dir):
                try:
                    logger.info(f"加载已有的 {doc_type} 向量库")
                    store = Chroma(
                        persist_directory=persist_dir,
                        embedding_function=embedding,
                        collection_name=doc_type
                    )
                    self.stores[doc_type] = store
                    return store
                except Exception as e:
                    logger.error(f"加载向量库失败: {e}")

        # 需要重建向量库
        return self.rebuild_store(doc_type, force)

    def rebuild_store(self, doc_type: str, force: bool = False):
        """重建向量库"""
        start_time = time.time()
        config = settings.chroma_collections[doc_type]
        persist_dir = os.path.join(settings.chroma_persist_dir, doc_type)
        embedding = get_embedding_model(config["embedding"])

        # 加载文档
        loader = DocumentLoader(doc_type)
        try:
            chunks = loader.get_processed_documents()
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            chunks = []

        if not chunks:
            logger.warning(f"没有找到文档片段: {doc_type}")

        # 创建向量存储
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_dir,
            collection_name=doc_type
        )

        # 更新状态
        self.stores[doc_type] = store
        self.needs_update[doc_type] = False

        logger.info(
            f"向量库重建完成: {doc_type} - {len(chunks)} 个文档片段 "
            f"(耗时: {time.time() - start_time:.2f}s)"
        )
        return store

    def get_store(self, doc_type: str):
        """获取向量存储实例"""
        if doc_type not in self.stores:
            return self.init_store(doc_type)

        # 检查是否需要更新
        if self.needs_update.get(doc_type, False) or needs_update(doc_type):
            return self.rebuild_store(doc_type)

        return self.stores[doc_type]

    def get_all_stores(self):
        """获取所有向量存储"""
        return {
            doc_type: self.get_store(doc_type)
            for doc_type in settings.chroma_collections
        }

    def get_document_count(self, doc_type: str) -> int:
        """获取文档片段数量"""
        store = self.get_store(doc_type)
        return store._collection.count() if store else 0


vector_store_service = VectorStoreService()