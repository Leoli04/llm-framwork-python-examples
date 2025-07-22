
from langchain_community.llms import Tongyi
from getpass import getpass
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

import os
import logging





def chain_PromptTemplate(llm):
    # 3. 创建提示模版
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)


    chain = prompt | llm

    # 3. 执行查询（添加异常处理）
    try:
        print("\n正在向大模型发送请求...")
        response = chain.invoke({"question": "langchain是什么？问题用question,回答用answer。"})
        print("\n=== 模型响应 ===")
        print(response)

    except Exception as e:
        print(f"\n⚠️ 请求失败: {str(e)}")
        print("可能原因:")
        print("- 无效的API密钥")
        print("- 未开通通义千问服务（需在阿里云控制台开通）")
        print("- 网络连接问题")
        print(f"- 账户余额不足（请访问: https://dashscope.console.aliyun.com/）")

def chain_ChatPromptTemplate(llm):
    # 3. 创建提示模版

    prompt = ChatPromptTemplate.from_messages([
        ("system","你是大模型应用架构师"),
        ("user","{input}")
    ])

    # output_parser = StrOutputParser()
    output_parser = JsonOutputParser()

    chain = prompt | llm | output_parser

    # 3. 执行查询（添加异常处理）
    try:
        print("\n正在向大模型发送请求...")
        response = chain.invoke({"input":"langchain是什么？问题用question,回答用answer。用json格式回复"})
        print("\n=== 模型响应 ===")
        print(response)

    except Exception as e:
        print(f"\n⚠️ 请求失败: {str(e)}")
        print("可能原因:")
        print("- 无效的API密钥")
        print("- 未开通通义千问服务（需在阿里云控制台开通）")
        print("- 网络连接问题")
        print(f"- 账户余额不足（请访问: https://dashscope.console.aliyun.com/）")


if __name__ == '__main__':
    # 配置日志（显示调试信息）
    logging.basicConfig(level=logging.INFO)

    # 1. 获取API密钥
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or getpass("请输入密钥：")
    print("DASHSCOPE_API_KEY:", DASHSCOPE_API_KEY)
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

    # 2. 初始化模型（添加超时和重试参数）
    llm = Tongyi(
        mo="qwen-plus",  # 明确指定模型
        # max_retries=3,  # 失败重试
        # request_timeout=30  # 30秒超时
    )

    # chain_PromptTemplate(llm)
    chain_ChatPromptTemplate(llm)
