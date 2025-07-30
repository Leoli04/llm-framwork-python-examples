from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessageChunk, HumanMessage
from typing import Annotated
from typing_extensions import TypedDict
import asyncio
from common_code import load_env, init_chat_model

import operator

# 定义state结构
class State(TypedDict):
    messages:Annotated[list,operator.add]

def weather_node(state:State):
    return {"messages": ["The weather is sunny and 25°C."]}

def calculator_node(state: State):
    return {"messages": ["The result of 2 + 2 is 4."]}


def simulate_interaction_with_full_state_stream(app,input_message):

    # Stream the full state of the graph
    for result in app.stream(input_message, stream_mode="values"):
        print(result)  # Print the full state after each node


def simulate_interaction_with_update_stream(app,input_message):

    # Stream updates after each node
    for result in app.stream(input_message, stream_mode="updates"):
        print(result)

async def simulate_interaction_with_token_stream(app,input_message):
    first = True
    # Stream LLM tokens
    async for msg, metadata  in app.astream(input_message, stream_mode="messages"):
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="|", flush=True)

        if isinstance(msg, AIMessageChunk):
            if first:
                gathered = msg
                first = False
            else:
                gathered = gathered + msg

            if msg.tool_call_chunks:
                print(gathered.tool_calls)

if __name__ == '__main__':
    workflow = StateGraph(State)
    workflow.add_node("weather_node", weather_node)
    workflow.add_node("calculator_node", calculator_node)

    workflow.add_edge(START, "weather_node")
    workflow.add_edge("weather_node", "calculator_node")
    workflow.add_edge("calculator_node", END)

    app = workflow.compile()

    input_message = {"messages": [("human", "Tell me the weather")]}

    simulate_interaction_with_full_state_stream(app,input_message)

    print("=========")
    simulate_interaction_with_update_stream(app,input_message)
    print("=========")

    # asyncio.run(simulate_interaction_with_token_stream(app,input_message))
