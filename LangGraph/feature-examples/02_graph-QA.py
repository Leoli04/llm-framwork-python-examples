from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from common_code import load_env, init_chat_model


def call_llm(state: MessagesState):
    '''
    使用大模型处理用户请求
    :param state:
    :return:
    '''
    messages = state["messages"]
    response = model.invoke(messages[-1].content)
    if response.tool_calls:
        tool_result = tool_node.invoke({"messages":[response]})
        tool_message = tool_result["messages"][-1].content
        response.content += f"\nTool Result: {tool_message}"
    return {"messages": [response]}


def interact_with_agent():
    '''
    与大模型互动
    :return:
    '''
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending the conversation.")
            break
        input_message = {
            "messages": [("human", user_input)]
        }

        for chunk in app.stream(input_message, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

@tool
def get_weather(location: str):
    """Fetch the current weather for a specific location."""
    weather_data = {
        "San Francisco": "It's 60 degrees and foggy.",
        "New York": "It's 90 degrees and sunny.",
        "London": "It's 70 degrees and cloudy.",
        "Nairobi": "It's 27 degrees celsius and sunny."
    }
    return weather_data.get(location, "Weather information is unavailable for this location.")



if __name__ == '__main__':

    load_env()

    # tools = [get_weather]
    # tool_node = ToolNode(tools)
    tool_node = ToolNode([get_weather], handle_tool_errors=False)

    model = init_chat_model("qwen-turbo").bind_tools([get_weather])

    # 定义图
    workflow = StateGraph(MessagesState)
    # 添加call_llm节点
    workflow.add_node("call_llm", call_llm)

    # 定义边(start -> LLM -> end)
    workflow.add_edge(START, "call_llm")
    workflow.add_edge("call_llm", END)

    # 编译工作流
    app = workflow.compile()

    interact_with_agent()

    # input_message = {
    #     "messages": [("human", "肯尼亚的首都是什么?")]
    # }
    #
    # for chunk in app.stream(input_message, stream_mode="values"):
    #     chunk["messages"][-1].pretty_print()


