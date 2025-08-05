from langchain_core.runnables import RunnableLambda
from app.config import settings
from app.services.vector_store import vector_store_service
from app.utils.logger import logger


class RetrieverService:
    def __init__(self):
        self.stores = vector_store_service.get_all_stores()

    def route_query(self, query: str) -> str:
        """智能路由查询到合适的文档集"""
        # 简化版路由逻辑 - 实际应用可用LLM优化
        query_lower = query.lower()

        # 产品文档关键词
        product_keywords = ["产品", "价格", "购买", "规格", "订单", "商品"]
        # 技术文档关键词
        tech_keywords = ["技术", "API", "错误", "代码", "配置", "安装", "文档"]

        if any(kw in query_lower for kw in product_keywords):
            return "product"
        elif any(kw in query_lower for kw in tech_keywords):
            return "technical"
        return "product"  # 默认

    def get_retriever(self, doc_type: str, top_k: int = None):
        """获取指定类型的检索器"""
        top_k = top_k or settings.default_top_k
        store = vector_store_service.get_store(doc_type)
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

    def hybrid_retrieve(self, query: str, top_k: int = None):
        """混合检索多个文档集"""
        top_k = top_k or settings.default_top_k
        all_results = []

        for doc_type, store in self.stores.items():
            try:
                results = store.similarity_search(
                    query,
                    k=top_k,
                    filter={"doc_type": doc_type}
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"检索失败 ({doc_type}): {e}")

        # 按相似度排序
        all_results.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
        return all_results[:top_k * 2]  # 返回双倍结果供LLM筛选

    def build_retrieval_chain(self):
        """构建带路由的检索链"""

        def route(inputs):
            query = inputs["query"]
            doc_type = self.route_query(query)
            return self.get_retriever(doc_type).invoke(query)

        return RunnableLambda(route)


retriever_service = RetrieverService()