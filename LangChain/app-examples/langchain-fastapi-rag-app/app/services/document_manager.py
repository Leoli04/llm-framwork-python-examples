import os
import shutil
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from app.config import settings
from app.services.vector_store import vector_store_service
from app.core.processing import process_document
from app.utils.logger import logger
from concurrent.futures import ThreadPoolExecutor


class DocumentEventHandler(FileSystemEventHandler):
    """监控文档目录变化"""

    def __init__(self, doc_type: str):
        self.doc_type = doc_type
        self.pending_dir = os.path.join(
            settings.chroma_collections[doc_type]["source_dir"],
            "pending"
        )
        self.processed_dir = os.path.join(
            settings.chroma_collections[doc_type]["source_dir"],
            "processed"
        )

    def on_created(self, event):
        if not event.is_directory:
            self.handle_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.handle_file(event.src_path)

    def handle_file(self, file_path: str):
        """处理文件变更"""
        if not os.path.isfile(file_path):
            return

        logger.info(f"检测到文档变更: {file_path}")

        # 异步处理文档
        with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
            executor.submit(self.process_document_async, file_path)

    def process_document_async(self, file_path: str):
        """异步处理文档"""
        try:
            # 处理文档
            processed = process_document(file_path, self.doc_type)

            if not processed:
                logger.error(f"文档处理失败: {file_path}")
                return False

            # 移动文件到已处理目录
            filename = os.path.basename(file_path)
            processed_path = os.path.join(self.processed_dir, filename)
            shutil.move(file_path, processed_path)

            # 标记向量库需要更新
            vector_store_service.mark_for_update(self.doc_type)

            logger.info(f"文档处理完成: {file_path} → {processed_path}")
            return True
        except Exception as e:
            logger.error(f"文档处理失败: {file_path} - {str(e)}")
            return False


class DocumentManager:
    """文档处理管理器"""

    def __init__(self):
        self.observers = {}
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self.running = False

    def start_watchers(self):
        """启动所有文档类型的监控"""
        if self.running:
            logger.warning("文档监控已在运行中")
            return

        for doc_type in settings.chroma_collections.keys():
            self.start_watcher(doc_type)

        self.running = True
        logger.info("所有文档监控已启动")

    def start_watcher(self, doc_type: str):
        """启动指定文档类型的监控"""
        config = settings.chroma_collections[doc_type]
        pending_dir = os.path.join(config["source_dir"], "pending")

        # 创建目录结构
        os.makedirs(pending_dir, exist_ok=True)
        os.makedirs(os.path.join(config["source_dir"], "processed"), exist_ok=True)

        # 启动文件监控
        event_handler = DocumentEventHandler(doc_type)
        observer = Observer()
        observer.schedule(event_handler, pending_dir, recursive=True)
        observer.start()

        self.observers[doc_type] = observer
        logger.info(f"启动文档监控: {doc_type} -> {pending_dir}")

    def stop_watchers(self):
        """停止所有监控"""
        for doc_type, observer in self.observers.items():
            observer.stop()
            observer.join()
            logger.info(f"停止文档监控: {doc_type}")

        self.observers = {}
        self.running = False
        logger.info("所有文档监控已停止")

    def process_pending_documents(self, doc_type: str):
        """处理所有待处理文档"""
        config = settings.chroma_collections[doc_type]
        pending_dir = os.path.join(config["source_dir"], "pending")

        if not os.path.exists(pending_dir):
            return []

        results = []
        for filename in os.listdir(pending_dir):
            file_path = os.path.join(pending_dir, filename)
            if os.path.isfile(file_path):
                result = self.executor.submit(
                    self.observers[doc_type].event_handler.process_document_async,
                    file_path
                )
                results.append(result)

        return results

    def manual_process(self, doc_type: str, file_path: str) -> dict:
        """手动处理文档"""
        if not os.path.isfile(file_path):
            return {"status": "error", "message": "文件不存在"}

        config = settings.chroma_collections[doc_type]
        pending_dir = os.path.join(config["source_dir"], "pending")
        os.makedirs(pending_dir, exist_ok=True)

        # 复制文件到待处理目录
        filename = os.path.basename(file_path)
        target_path = os.path.join(pending_dir, filename)
        shutil.copy(file_path, target_path)

        logger.info(f"已提交手动处理: {file_path} -> {target_path}")
        return {
            "status": "submitted",
            "path": target_path,
            "doc_type": doc_type
        }

    def bulk_import(self, doc_type: str, directory: str) -> dict:
        """批量导入文档"""
        if not os.path.isdir(directory):
            return {"status": "error", "message": "目录不存在"}

        config = settings.chroma_collections[doc_type]
        pending_dir = os.path.join(config["source_dir"], "pending")
        os.makedirs(pending_dir, exist_ok=True)

        count = 0
        for filename in os.listdir(directory):
            src_path = os.path.join(directory, filename)
            if os.path.isfile(src_path):
                target_path = os.path.join(pending_dir, filename)
                shutil.copy(src_path, target_path)
                count += 1

        logger.info(f"批量导入完成: {count} 个文档 -> {doc_type}")
        return {"status": "submitted", "count": count, "doc_type": doc_type}


document_manager = DocumentManager()