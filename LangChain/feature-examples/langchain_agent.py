
# TavilySearchResults将在 LangChain 1.0 中被彻底移除
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv


if __name__ == '__main__':
    # 加载.env文件中的变量
    load_dotenv()

    # 定义搜索引擎工具
    search = TavilySearch(max_results=2)

    # 将工具添加到list
    tools = [search]

    # 2. 初始化模型
    model = ChatTongyi(
        model="qwen-turbo",  # 明确指定模型
    )

    memory = MemorySaver()

    # response = model.invoke([HumanMessage(content="你好!")])

    # # 将工具绑定到模型上
    model_with_tools = model.bind_tools(tools)
    #
    # response = model_with_tools.invoke([HumanMessage(content="你好!")])
    #
    # print(f"ContentString: {response.content}")
    # print(f"ToolCalls: {response.tool_calls}")

    # 创建代理，
    '''
    传入的是 model，而不是 model_with_tools。这是因为 create_react_agent 会在后台为我们调用 .bind_tools
    '''
    agent_executor = create_react_agent(model, tools,checkpointer=memory)

    # response = agent_executor.invoke({"messages": [HumanMessage(content="上海天气怎么样!")]})
    #
    # print(response["messages"])

    config = {"configurable": {"thread_id": "abc123"}}
    #     使用流式消息
    for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content="你好，我是leo! 我住在上海")]}, config
    ):
        print(chunk)
        print("----")

    print("=====================")
    for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content="我住的地方天气怎么样?")]}, config
    ):
        print(chunk)
        print("----")
