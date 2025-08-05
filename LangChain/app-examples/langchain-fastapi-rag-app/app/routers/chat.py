from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm_service import llm_service
import time

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    session_id: str = None


class ChatResponse(BaseModel):
    answer: str
    sources: list = []
    latency: float


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """问答API端点"""
    start_time = time.time()

    # 执行查询
    answer = await llm_service.query(request.question)

    # 构造响应
    latency = time.time() - start_time
    return ChatResponse(
        answer=answer,
        sources=[],  # 实际可添加来源文档
        latency=round(latency, 3)
    )