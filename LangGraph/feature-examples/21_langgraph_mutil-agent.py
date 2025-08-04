import operator
import os
from functools import partial
from typing import Literal, TypedDict, Annotated, Sequence

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from common_code import load_env, init_chat_model
from display_graph import display_graph

# Define RouteResponse for Customer Service Supervisor
class RouteResponseCS(BaseModel):
    next: Literal["Query_Agent", "Resolution_Agent", "Escalation_Agent", "FINISH"]

def supervisor_agent_cs(state):
    supervisor_chain_cs = prompt_cs | llm.with_structured_output(RouteResponseCS)
    return supervisor_chain_cs.invoke(state)

# Agent node function to handle message flow to each agent
def agent_node(state, agent, name):
    '''
    通用代理执行节点
    :param state:
    :param agent:
    :param name:
    :return:
    '''
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

# Define Customer Service graph state and workflow
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def get_wordflow_graph():
    # Initialize StateGraph and add nodes
    workflow_cs = StateGraph(AgentState)
    workflow_cs.add_node("Query_Agent", query_node)
    workflow_cs.add_node("Resolution_Agent", resolution_node)
    workflow_cs.add_node("Escalation_Agent", escalation_node)
    workflow_cs.add_node("supervisor", supervisor_agent_cs)

    # Define edges for agents to return to the supervisor
    for member in members_cs:
        workflow_cs.add_edge(member, "supervisor")

    # Define conditional map for routing
    conditional_map_cs = {k: k for k in members_cs}
    conditional_map_cs["FINISH"] = END
    workflow_cs.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map_cs)
    workflow_cs.add_edge(START, "supervisor")

if __name__ == '__main__':
    '''
    用户服务场景中的多代理场景
    '''

    load_env()

    # step 1: 定义主管代理节点
    members_cs = ["Query_Agent", "Resolution_Agent", "Escalation_Agent"]
    # supervisor_agent_llm = get_supervisor_agent_llm()
    system_prompt_cs = f"You are a Customer Service Supervisor managing agents: {', '.join(members_cs)}."
    # Create prompt template for the supervisor with correctly formatted options
    prompt_cs = ChatPromptTemplate.from_messages([
        ("system", system_prompt_cs),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Choose the next agent to act from {options}."),
    ]).partial(options=str(members_cs))
    llm = init_chat_model()

    # step 2: 定义执行具体任务的节点
    # Define agents for Customer Service tasks with realistic tools
    query_agent = create_react_agent(llm, tools=[TavilySearchResults(max_results=5)])
    resolution_agent = create_react_agent(llm, tools=[PythonREPLTool()])
    escalation_agent = create_react_agent(llm, tools=[PythonREPLTool()])

    # Create nodes for each agent with valid names
    query_node = partial(agent_node, agent=query_agent, name="Query_Agent")
    resolution_node = partial(agent_node, agent=resolution_agent, name="Resolution_Agent")
    escalation_node = partial(agent_node, agent=escalation_agent, name="Escalation_Agent")

    # step 3: 定义图
    graph_cs = get_wordflow_graph()

    # Display the graph
    display_graph(graph_cs, file_name=os.path.basename(__file__))

    # Example input for testing
    inputs_cs = {"messages": [HumanMessage(content="Help me reset my password.")]}

    # Run the graph
    for output in graph_cs.stream(inputs_cs):
        if "__end__" not in output:
            print(output)
