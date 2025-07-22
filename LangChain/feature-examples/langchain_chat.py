from langchain_community.llms import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from getpass import getpass

import os
import logging

# 配置日志（显示调试信息）
logging.basicConfig(level=logging.INFO)

# 1. 获取API密钥qwen-turbo
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or getpass("请输入密钥：")
print("DASHSCOPE_API_KEY:",DASHSCOPE_API_KEY)
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 2. 初始化模型（添加超时和重试参数）
model = ChatTongyi(
    model="",  # 明确指定模型
    # top_p
    max_retries=3  # 失败重试
    # streaming
    # api_key

)

from langchain_core.messages import HumanMessage



# 3. 执行查询（添加异常处理）
try:
    print("正在向通义千问发送请求...")
    response = model.invoke([HumanMessage(content="我是leo")])
    print("\n=== 模型响应 ===")
    print(response)

    print("\n=== 再次询问 ===")
    response = model.invoke([HumanMessage(content="我是谁")])
    print("\n=== 模型响应 ===")
    print(response)


except Exception as e:
    print(f"\n⚠️ 请求失败: {str(e)}")
    print("可能原因:")
    print("- 无效的API密钥")
    print("- 未开通通义千问服务（需在阿里云控制台开通）")
    print("- 网络连接问题")
    print(f"- 账户余额不足（请访问: https://dashscope.console.aliyun.com/）")

