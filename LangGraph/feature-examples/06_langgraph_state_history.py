from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict


class State(TypedDict):
    foo: str
    bar: list[str]


def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


if __name__ == '__main__':
    workflow = StateGraph(State)
    workflow.add_node(node_a)
    workflow.add_node(node_b)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"foo": "", "bar": []}, config)

    latest_state = graph.get_state(config)
    print(latest_state.values)

    config = {"configurable": {"thread_id": "1"}}
    state_history = graph.get_state_history(config)
    for snapshot in state_history:
        print(snapshot.values)
