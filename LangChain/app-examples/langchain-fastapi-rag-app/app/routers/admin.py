from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.document_manager import document_manager
from app.services.vector_store import vector_store_service
import os
import uuid
import shutil

router = APIRouter()


@router.post("/documents/upload")
async def upload_document(
        doc_type: str = Form(...),
        file: UploadFile = File(...)
):
    """上传并处理文档"""
    # 创建临时目录
    temp_dir = f"temp/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    # 保存文件
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(500, f"文件保存失败: {str(e)}")

    # 提交处理
    try:
        result = document_manager.manual_process(doc_type, temp_path)
        return result
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/documents/rebuild/{doc_type}")
async def rebuild_vector_store(doc_type: str):
    """手动重建向量库"""
    try:
        vector_store_service.mark_for_update(doc_type)
        vector_store_service.rebuild_store(doc_type, force=True)
        return {"status": "success", "message": f"{doc_type} 向量库已重建"}
    except Exception as e:
        raise HTTPException(500, f"重建失败: {str(e)}")


@router.post("/documents/bulk-import")
async def bulk_import_documents(
        doc_type: str = Form(...),
        zip_file: UploadFile = File(...)
):
    """批量导入文档（ZIP格式）"""
    # 创建临时目录
    temp_dir = f"temp/bulk_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, zip_file.filename)

    # 保存ZIP文件
    try:
        with open(zip_path, "wb") as buffer:
            content = await zip_file.read()
            buffer.write(content)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(500, f"文件保存失败: {str(e)}")

    # 解压ZIP文件
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(500, f"解压失败: {str(e)}")

    # 批量导入
    try:
        result = document_manager.bulk_import(doc_type, temp_dir)
        return result
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/status")
def system_status():
    """获取系统状态"""
    status = {
        "vector_stores": {},
        "document_manager": document_manager.running
    }

    for doc_type in vector_store_service.stores.keys():
        status["vector_stores"][doc_type] = {
            "document_count": vector_store_service.get_document_count(doc_type),
            "needs_update": vector_store_service.needs_update.get(doc_type, False)
        }

    return status