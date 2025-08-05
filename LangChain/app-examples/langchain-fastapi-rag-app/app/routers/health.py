from fastapi import APIRouter
from app.services.vector_store import vector_store_service
from app.utils.logger import logger

router = APIRouter()


@router.get("/live")
def liveness_check():
    """存活检查"""
    return {"status": "alive"}


@router.get("/ready")
def readiness_check():
    """就绪检查"""
    try:
        # 检查所有向量库
        for doc_type in vector_store_service.stores.keys():
            count = vector_store_service.get_document_count(doc_type)
            if count == 0:
                logger.warning(f"向量库 {doc_type} 无文档")
                return {"status": "not_ready", "reason": f"{doc_type} 向量库为空"}

        return {"status": "ready"}
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        return {"status": "not_ready", "reason": str(e)}