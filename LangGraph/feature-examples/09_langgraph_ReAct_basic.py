from common_code import load_env, init_chat_model
from langgraph.prebuilt import create_react_agent

from display_graph import display_graph

import os

# Define tools
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

if __name__ == '__main__':

    load_env()
    model = init_chat_model()

    tools = [add, multiply]

    agent = create_react_agent(model=model,tools=tools)

    # Visualise the graph
    # display_graph(agent, file_name=os.path.basename(__file__))

    # User input
    inputs = {"messages": [("user", "Add 32 and 4. Multiply the result by 2 and divide by 4.")]}

    # Run the ReAct agent
    messages = agent.invoke(inputs)
    for message in messages["messages"]:
        message.pretty_print()

