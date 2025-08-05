import schedule
import threading
import time
from app.services.vector_store import vector_store_service
from app.utils.logger import logger
from app.core.versioning import needs_update
from chromadb.app import settings


class BackgroundScheduler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()

    def run(self):
        """定时任务调度"""
        # 每晚2点执行向量库完整性检查
        schedule.every().day.at("02:00").do(self.check_vector_store_integrity)

        # 每30分钟检查文档更新
        schedule.every(30).minutes.do(self.check_document_updates)

        logger.info("后台定时任务已启动")

        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)

    def stop(self):
        self.stop_event.set()
        logger.info("后台定时任务已停止")

    def check_vector_store_integrity(self):
        """检查向量库完整性"""
        logger.info("开始向量库完整性检查")
        for doc_type in vector_store_service.stores.keys():
            try:
                # 简单验证：获取文档数量
                count = vector_store_service.get_document_count(doc_type)
                logger.info(f"向量库 {doc_type} 完整性检查: {count} 个文档")
            except Exception as e:
                logger.error(f"向量库 {doc_type} 检查失败: {str(e)}")
                vector_store_service.mark_for_update(doc_type)

    def check_document_updates(self):
        """检查文档更新（外部修改）"""
        logger.info("开始文档更新检查")
        for doc_type in settings.chroma_collections.keys():
            if needs_update(doc_type):
                logger.info(f"检测到文档更新: {doc_type}")
                vector_store_service.mark_for_update(doc_type)


# 在应用启动时启动定时任务
scheduler = BackgroundScheduler()