# from langgraph.constants import START, END
import os
import random
import subprocess
import sys

# from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from  display_graph import display_graph


# 定义state结构
class HelloWorldState(TypedDict):
    greeting: str


# 定义node 方法
def hello_world_node(state: HelloWorldState):
    state["greeting"] = "Hello World, " + state["greeting"]
    return state


def exclamation_node(state: HelloWorldState):
    state["greeting"] += "!"
    return state


# 初始化图，并添加node
builder = StateGraph(HelloWorldState)
# # 将hello_world_node函数作为图中的一个节点，标记为“greet”
builder.add_node("greet", hello_world_node)
builder.add_node("exclaim", exclamation_node)

# 定义边
builder.add_edge(START, "greet")
builder.add_edge("greet", "exclaim")
builder.add_edge("exclaim", END)

# 编译并运行
graph = builder.compile()
result = graph.invoke({"greeting": "from LangGraph!"})

print(result)

# 将graph可视化
display_graph(graph,"01")
