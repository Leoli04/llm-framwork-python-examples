


from typing import TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from common_code import load_env,init_chat_model

# Define the agent's state
class AgentState(TypedDict):
    query: str
    response: str

load_env()
llm_tool = init_chat_model()


# Define the node that processes user queries
def handle_query(state: AgentState) -> AgentState:
    user_message = HumanMessage(content=state['query'])
    ai_response = llm_tool.invoke([user_message])
    state['response'] = ai_response.content
    return state

# Build the LangGraph workflow for the agent
builder = StateGraph(AgentState)
builder.add_node("handle_query", handle_query)
builder.add_edge(START, "handle_query")
builder.add_edge("handle_query", END)
graph = builder.compile()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str


@app.post("/api/research")
async def research_query(request: QueryRequest):
    try:

        initial_state = {"query": request.query, "response": ""}
        result = graph.invoke(initial_state)
        return {"response": result['response']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/stream")
async def research_query_stream(request: QueryRequest):
    async def event_generator():
        try:
            initial_state = {"query": request.query, "response": ""}
            async for msg, metadata in graph.astream(initial_state, stream_mode="messages"):
                if msg.content and not isinstance(msg, HumanMessage):
                    yield msg.content
        except Exception as e:
            yield {"error": str(e)}

    return StreamingResponse(event_generator(), media_type="text/plain")

#RUN THE APP
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)