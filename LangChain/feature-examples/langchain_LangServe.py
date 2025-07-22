
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Tongyi
from langserve import add_routes

# 1. 创建提示词模版
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. 初始化模型
model = Tongyi(
        model_name="qwen-plus",  # 明确指定模型
        # max_retries=3,  # 失败重试
        # request_timeout=30  # 30秒超时
    )

# 3. 输出解析
parser = StrOutputParser()

# 4. 使用LCEL连接
chain = prompt_template | model | parser


# 4. app定义
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. 添加chain路由
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)