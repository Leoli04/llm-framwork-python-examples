from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict

from common_code import load_env, init_chat_model

import os


# 定义代理和子图的共享数据状态
class ReActAgentState(TypedDict):
    message: str  # Shared key between the parent graph and sub-graphs
    action: str   # Sub-graph specific key (what action to perform)
    sub_action: str  # Additional sub-action to perform in a more complex scenario

# 推理节点 1:根据用户问题决定执行哪个代理
def reasoning_node(state: ReActAgentState):
    query = state["message"]
    if "weather" in query:
        return {"action": "fetch_weather"}
    elif "news" in query:
        return {"action": "fetch_news"}
    elif "recommend" in query:
        return {"action": "recommendation", "sub_action": "book"}
    else:
        return {"action": "unknown"}

# Sub-graph for fetching weather information (acting phase)
def weather_subgraph_node(state: ReActAgentState):
    # Simulating a weather tool call
    return {"message": "The weather today is sunny."}

# Sub-graph for fetching news information (acting phase)
def news_subgraph_node(state: ReActAgentState):
    # Simulating a news tool call
    return {"message": "Here are the latest news headlines."}

# Sub-graph for providing a book recommendation (acting phase)
def recommendation_subgraph_node(state: ReActAgentState):
    if state.get("sub_action") == "book":
        return {"message": "I recommend reading 'The Pragmatic Programmer'."}
    else:
        return {"message": "I have no other recommendations at the moment."}

# Define dynamic reasoning node in the parent graph to route to the correct sub-graph
def reasoning_state_manager(state: ReActAgentState):
    if state["action"] == "fetch_weather":
        return weather_subgraph
    elif state["action"] == "fetch_news":
        return news_subgraph
    elif state["action"] == "recommendation":
        return recommendation_subgraph
    else:
        return None

if __name__ == '__main__':
    # Build sub-graph for fetching weather information
    weather_subgraph_builder = StateGraph(ReActAgentState)
    weather_subgraph_builder.add_node("weather_action", weather_subgraph_node)
    weather_subgraph_builder.set_entry_point("weather_action")
    weather_subgraph = weather_subgraph_builder.compile()

    # Build sub-graph for fetching news information
    news_subgraph_builder = StateGraph(ReActAgentState)
    news_subgraph_builder.add_node("news_action", news_subgraph_node)
    news_subgraph_builder.set_entry_point("news_action")
    news_subgraph = news_subgraph_builder.compile()

    # Build sub-graph for recommendations (e.g., book recommendation)
    recommendation_subgraph_builder = StateGraph(ReActAgentState)
    recommendation_subgraph_builder.add_node("recommendation_action", recommendation_subgraph_node)
    recommendation_subgraph_builder.set_entry_point("recommendation_action")
    recommendation_subgraph = recommendation_subgraph_builder.compile()

    # Create the parent graph
    parent_builder = StateGraph(ReActAgentState)
    parent_builder.add_node("reasoning", reasoning_node)
    parent_builder.add_node("action_dispatch", reasoning_state_manager)

    # Define edges in the parent graph
    parent_builder.add_edge(START, "reasoning")
    parent_builder.add_edge("reasoning", "action_dispatch")

    # Compile the parent graph
    react_agent_graph = parent_builder.compile()

    # Test the agent with a weather-related query
    inputs_weather = {"message": "What is the weather today?"}
    result_weather = react_agent_graph.invoke(inputs_weather)
    print(result_weather["message"])

    # Test the agent with a news-related query
    inputs_news = {"message": "Give me the latest news."}
    result_news = react_agent_graph.invoke(inputs_news)
    print(result_news["message"])

    # Test the agent with a recommendation-related query
    inputs_recommendation = {"message": "Can you recommend a good book?"}
    result_recommendation = react_agent_graph.invoke(inputs_recommendation)
    print(result_recommendation["message"])

