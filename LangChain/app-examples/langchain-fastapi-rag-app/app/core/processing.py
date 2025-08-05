import os
import time
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.utils.logger import logger
from app.core.versioning import calculate_file_hash

# 支持的文档类型
SUPPORTED_EXTENSIONS = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.md': UnstructuredMarkdownLoader,
    '.pptx': UnstructuredPowerPointLoader,
    '.xlsx': UnstructuredExcelLoader
}


def get_file_loader(file_path: str):
    """根据文件扩展名获取合适的加载器"""
    ext = os.path.splitext(file_path)[1].lower()
    loader_class = SUPPORTED_EXTENSIONS.get(ext)
    if not loader_class:
        raise ValueError(f"不支持的文档格式: {ext}")
    return loader_class(file_path)


def process_document(file_path: str, doc_type: str):
    """处理单个文档的完整流水线"""
    logger.info(f"开始处理文档: {file_path}")
    start_time = time.time()

    try:
        # 1. 加载文档
        loader = get_file_loader(file_path)
        documents = loader.load()

        # 2. 文档分割
        config = settings.chroma_collections[doc_type]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(documents)

        # 3. 添加元数据
        filename = os.path.basename(file_path)
        file_hash = calculate_file_hash(file_path)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "doc_id": f"{filename}_{i}",
                "doc_type": doc_type,
                "source": filename,
                "file_path": file_path,
                "file_hash": file_hash,
                "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        logger.info(f"文档处理完成: {file_path} → {len(chunks)} 个片段 "
                    f"(耗时: {time.time() - start_time:.2f}s)")
        return chunks
    except Exception as e:
        logger.error(f"文档处理失败: {file_path} - {str(e)}")
        raise