import os
import json
import hashlib
import time

from app.config import settings
from app.utils.logger import logger


def calculate_file_hash(file_path: str) -> str:
    """计算文件内容哈希值"""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"计算文件哈希失败: {file_path} - {e}")
        return ""


def get_version_file_path(doc_type: str) -> str:
    """获取版本文件路径"""
    return os.path.join(
        settings.chroma_collections[doc_type]["source_dir"],
        "processed",
        ".version.json"
    )


def get_current_version(doc_type: str) -> dict:
    """获取当前版本信息"""
    version_file = get_version_file_path(doc_type)
    if os.path.exists(version_file):
        try:
            with open(version_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取版本文件失败: {version_file} - {e}")
    return {}


def update_version_file(doc_type: str, file_hashes: dict):
    """更新版本文件"""
    version_file = get_version_file_path(doc_type)
    version_data = {
        "file_hashes": file_hashes,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open(version_file, "w") as f:
            json.dump(version_data, f, indent=2)
        logger.info(f"版本文件更新: {version_file}")
    except Exception as e:
        logger.error(f"更新版本文件失败: {version_file} - {e}")


def needs_update(doc_type: str) -> bool:
    """检查文档是否需要更新"""
    processed_dir = os.path.join(
        settings.chroma_collections[doc_type]["source_dir"],
        "processed"
    )

    # 获取当前版本
    current_version = get_current_version(doc_type)
    current_hashes = current_version.get("file_hashes", {})

    # 检查每个文件
    for filename in os.listdir(processed_dir):
        if filename == ".version.json":
            continue

        file_path = os.path.join(processed_dir, filename)
        if not os.path.isfile(file_path):
            continue

        # 计算当前哈希
        current_hash = calculate_file_hash(file_path)

        # 比较版本
        if filename not in current_hashes or current_hashes[filename] != current_hash:
            logger.info(f"检测到文件变更: {filename}")
            return True

    return False