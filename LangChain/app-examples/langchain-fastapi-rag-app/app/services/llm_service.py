import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.config import settings
from app.services.retriever import retriever_service
from app.utils.logger import logger


class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.3
        )
        self.qa_chain = self.build_qa_chain()

    def format_docs(self, docs):
        """格式化检索到的文档"""
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content
            formatted.append(f"【来源: {source}】\n{content}")
        return "\n\n".join(formatted)

    def build_qa_chain(self):
        """构建问答链"""
        prompt = ChatPromptTemplate.from_template("""
        你是一个专业的客服助手，请根据以下上下文回答问题：
        {context}

        问题：{question}

        要求：
        1. 回答要简洁专业
        2. 如果上下文不包含答案，请回答"根据已知信息无法回答该问题"
        3. 用中文回答
        """)

        return (
                {"context": retriever_service.build_retrieval_chain() | self.format_docs,
                 "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

    async def query(self, question: str) -> str:
        """执行问答查询"""
        try:
            start_time = time.time()
            response = await self.qa_chain.ainvoke(question)
            latency = time.time() - start_time

            logger.info(f"问答完成: 问题={question[:50]}... "
                        f"耗时={latency:.2f}s 字数={len(response)}")
            return response
        except Exception as e:
            logger.error(f"问答失败: {question} - {e}")
            return "抱歉，处理您的请求时出错了"


llm_service = LLMService()