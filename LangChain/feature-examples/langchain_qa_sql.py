from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage

import os


def getDbByOs():
    # 获取当前 Python 文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建相对路径：从 python/ 目录向上退两级到 langchain/
    db_relative_path = os.path.join(current_dir, '..', 'Chinook.db')

    # 转换为绝对路径并规范化
    db_absolute_path = os.path.abspath(db_relative_path)

    print("db_absolute_path:", db_absolute_path)
    # 创建数据库连接
    uri = f"sqlite:///{db_absolute_path}"
    return SQLDatabase.from_uri(uri)


def get_db_by_path():
    from pathlib import Path

    # 使用 pathlib 处理路径
    current_dir = Path(__file__).resolve().parent
    db_path = current_dir.parent / "Chinook.db"
    # 创建数据库连接
    # 创建 URI (Windows 需要特殊处理)
    if os.name == 'nt':  # Windows 系统
        uri = f"sqlite:///{db_path.resolve()}"
    else:  # Linux/Mac
        uri = f"sqlite:////{db_path.resolve()}"

    return SQLDatabase.from_uri(uri)


def load_env():
    from dotenv import load_dotenv
    load_dotenv()  # 自动加载 .env 文件


def init_chat_model(model_name: str):
    from langchain_community.chat_models import ChatTongyi

    return ChatTongyi(model=model_name)


# 添加一个函数来提取纯 SQL 语句
def extract_sql(response):
    # 如果响应是字符串且包含 "SQLQuery:"，则提取 SQL 部分
    if isinstance(response, str) and "SQLQuery:" in response:
        return response.split("SQLQuery:", 1)[1].strip()
    return response


# 创建打印中间结果的函数
def print_intermediate(data):
    print("\n=== 中间结果 ===")
    print(f"类型: {type(data)}")
    print(f"内容: {data}\n")
    return data


# 通过链的方式实现 sql查询，回答问题
def chat_with_sql_chain():
    from langchain.chains.sql_database.query import create_sql_query_chain

    # chain = create_sql_query_chain(model, db)
    # response = chain.invoke({"question": "有多少员工"})
    # print(response)
    # # SQLQuery: SELECT COUNT(*) AS "EmployeeCount" FROM "Employee"
    # print(db.run(response["SQLQuery"]))

    from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
    from langchain.schema.runnable import RunnableLambda

    execute_query = QuerySQLDatabaseTool(db=db)
    write_query = create_sql_query_chain(model, db)
    # chain = write_query | RunnableLambda(extract_sql) | execute_query
    # response = chain.invoke({"question": "有多少员工"})
    # print(response)

    from operator import itemgetter

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    '''

    '''
    chain = (
            RunnablePassthrough.assign(query=write_query)  # 使用 write_query（SQL 生成链）生成 SQL 查询
            | RunnableLambda(print_intermediate)  # 添加打印步骤
            .assign(result=itemgetter("query")  # itemgetter("query") 提取上一步生成的 SQL 查询
                           | RunnableLambda(print_intermediate)
                           | RunnableLambda(extract_sql)  # 去掉生成的sql部分的 SQLQuery:
                           | RunnableLambda(print_intermediate)
                           | execute_query)  # 执行sql查询
            | answer_prompt
            | model
            | StrOutputParser()
    )

    return chain


def chat_with_sql_agent():
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import create_react_agent
    from langchain_core.prompts import ChatPromptTemplate

    toolkit = SQLDatabaseToolkit(db=db, llm=model)

    tools = toolkit.get_tools()
    # print(tools)

    SQL_PREFIX = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=SQL_PREFIX)])

    agent_executor = create_react_agent(model, tools, prompt=prompt)

    return agent_executor


if __name__ == '__main__':
    load_env()
    db = get_db_by_path()
    # print(db.dialect)
    # print(db.get_usable_table_names())
    # print(db.run("SELECT * FROM Artist LIMIT 10;"))

    model = init_chat_model("qwen-plus")

    # chain = chat_with_sql_chain()
    # response = chain.invoke({"question": "有多少员工"})
    # print(response)

    agent_executor = chat_with_sql_agent()

    for s in agent_executor.stream(
            {"messages": [HumanMessage(content="有多少员工?")]}
    ):
        print(s)
        print("----")
