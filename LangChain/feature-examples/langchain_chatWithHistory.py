from langchain_community.llms import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from getpass import getpass

import os
import logging

# 配置日志（显示调试信息）
logging.basicConfig(level=logging.INFO)

# 1. 获取API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or getpass("请输入密钥：")
print("DASHSCOPE_API_KEY:",DASHSCOPE_API_KEY)
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 2. 初始化模型（添加超时和重试参数）
model = ChatTongyi(
    model="qwen-turbo",  # 明确指定模型
    # top_p
    max_retries=3  # 失败重试
    # streaming
    # api_key

)

from langchain_core.messages import HumanMessage

store = {}

def get_session_history(session_id: str)-> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model,get_session_history)

config = {"configurable": {"session_id": "abc2"}}



# 3. 执行查询（添加异常处理）
try:
    print("正在向通义千问发送请求...")
    response = with_message_history.invoke(
        [HumanMessage(content="我是leo")],
        config=config
    )
    print("\n=== 模型响应 ===")
    print(response)

    print("\n=== 再次询问 ===")
    response = with_message_history.invoke(
        [HumanMessage(content="我是谁")],
        config=config
    )
    print("\n=== 模型响应 ===")
    print(response)


except Exception as e:
    print(f"\n⚠️ 请求失败: {str(e)}")
    print("可能原因:")
    print("- 无效的API密钥")
    print("- 未开通通义千问服务（需在阿里云控制台开通）")
    print("- 网络连接问题")
    print(f"- 账户余额不足（请访问: https://dashscope.console.aliyun.com/）")

