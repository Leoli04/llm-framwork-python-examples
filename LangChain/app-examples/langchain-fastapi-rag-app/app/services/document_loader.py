import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.utils.logger import logger
from app.core.versioning import calculate_file_hash


class DocumentLoader:
    def __init__(self, doc_type: str):
        self.doc_type = doc_type
        self.config = settings.chroma_collections[doc_type]

    def load_documents_from_dir(self, directory: str):
        """从指定目录加载并分割文档"""
        loader = DirectoryLoader(
            directory,
            glob="**/*.*",
            show_progress=True,
            use_multithreading=True
        )
        docs = loader.load()

        # 文档分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=200
        )
        return text_splitter.split_documents(docs)

    def get_processed_documents(self):
        """获取已处理目录中的所有文档片段"""
        processed_dir = os.path.join(self.config["source_dir"], "processed")
        return self.load_documents_from_dir(processed_dir)

    def get_file_hashes(self):
        """获取所有已处理文件的哈希值"""
        processed_dir = os.path.join(self.config["source_dir"], "processed")
        hashes = {}

        for filename in os.listdir(processed_dir):
            if filename.startswith("."):
                continue

            file_path = os.path.join(processed_dir, filename)
            if os.path.isfile(file_path):
                hashes[filename] = calculate_file_hash(file_path)

        return hashes