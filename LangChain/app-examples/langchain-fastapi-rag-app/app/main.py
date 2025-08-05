import os
import asyncio
import time
import uvicorn
from chromadb.app import settings
from fastapi import FastAPI
from app.utils.logger import logger
from app.routers import chat, admin, health
from app.services.document_manager import document_manager
from app.services.vector_store import vector_store_service
from app.utils.scheduler import scheduler

app = FastAPI(
    title="RAG API 服务",
    description="基于多文档集的智能问答系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 包含路由
app.include_router(chat.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1/admin")
app.include_router(health.router, prefix="/health")


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("应用启动中...")
    start_time = time.time()

    try:
        # 1. 启动文档监控
        document_manager.start_watchers()

        # 2. 处理所有待处理文档
        processing_tasks = []
        for doc_type in settings.chroma_collections.keys():
            tasks = document_manager.process_pending_documents(doc_type)
            processing_tasks.extend(tasks)

        # 3. 等待文档处理完成（最多60秒）
        if processing_tasks:
            logger.info(f"等待 {len(processing_tasks)} 个文档处理完成...")
            wait_start = time.time()
            while not all(task.done() for task in processing_tasks):
                if time.time() - wait_start > 60:
                    logger.warning("文档处理超时，继续启动...")
                    break
                await asyncio.sleep(1)

        # 4. 初始化向量存储（并行）
        init_tasks = []
        for doc_type in settings.chroma_collections.keys():
            # 仅初始化标记为需要更新的向量库
            if vector_store_service.needs_update.get(doc_type, True):
                init_tasks.append(
                    asyncio.create_task(vector_store_service.init_store(doc_type))
                )

        if init_tasks:
            await asyncio.gather(*init_tasks)

        # 5. 启动定时任务
        scheduler.start()

        logger.info(f"应用启动完成 (耗时: {time.time() - start_time:.2f}s)")
    except Exception as e:
        logger.critical(f"应用启动失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    logger.info("应用关闭中...")

    try:
        # 停止文档监控
        document_manager.stop_watchers()

        # 停止定时任务
        scheduler.stop()

        logger.info("应用已安全关闭")
    except Exception as e:
        logger.error(f"关闭过程中出错: {e}")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)