from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage,ToolMessage

from common_code import load_env

# 定义工具
def get_user_profile(user_id: str):
    '''
    通过userId获取用户信息
    :param user_id:
    :return:
    '''
    user_data = {
        "101": {"name": "Alice", "age": 30, "location": "New York"},
        "102": {"name": "Bob", "age": 25, "location": "San Francisco"}
    }

    return user_data.get(user_id,"User profile not found.")

if __name__ == '__main__':

    # 加载环境变量
    load_env()

    # 设置工具节点
    tools = [get_user_profile]
    tool_node = ToolNode(tools)

    message_with_tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "get_user_profile",
            "args": {"user_id": "101"},
            "id": "tool_call_id",
            "type": "tool_call"
        }]
    )

    state = {
        "messages": [message_with_tool_call]
    }

    result = tool_node.invoke(state)

    # Output the result
    print(result)
