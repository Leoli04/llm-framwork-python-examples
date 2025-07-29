from common_code import load_env,init_chat_model

from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.checkpoint.memory import MemorySaver

import os

def call_llm(state:MessagesState):
    messages  = state["messages"]
    response = model.invoke(messages)
    return {"messages":[response]}

def interact_with_agent_with_memory():
    while True:
        thread_id = input("Enter thread ID (or 'new' for a new session): ")
        if thread_id.lower() in ["exit","quit"]:
            print("Ending the conversation.")
            break

        if thread_id.lower().strip() == "new":
            thread_id = f"session_{os.urandom(4).hex()}"

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "end session"]:
                print(f"Ending session {thread_id}.")
                break

            input_message = {
                "messages":[("human", user_input)]
            }

            config = {"configurable": {"thread_id": thread_id}}
            for chunk in app_with_memory.stream(input_message, config=config, stream_mode="values"):
                chunk["messages"][-1].pretty_print()

if __name__ == '__main__':
    # 加载环境变量
    load_env()

    # 初始化模型
    model = init_chat_model("qwen-turbo")

    workflow = StateGraph(MessagesState)

    workflow.add_node("call_llm", call_llm)
    workflow.add_edge(START, "call_llm")
    workflow.add_edge("call_llm", END)

    checkpointer = MemorySaver()

    app_with_memory = workflow.compile(checkpointer=checkpointer)

    interact_with_agent_with_memory()

