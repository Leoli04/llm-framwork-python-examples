

from typing import TypedDict

import socketio
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import emit, SocketIO
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,START,END

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

# 初始化 Flask 应用.创建一个 Flask 应用实例，`__name__` 是当前模块的名称，Flask 使用它来确定应用的位置。
app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
#
'''
配置 CORS 
Flask 的 CORS 配置（使用 `flask_cors`）只影响 HTTP 请求（如 AJAX），而 WebSocket 的跨域设置由 SocketIO 单独处理。因此，这里有两个地方配置了跨域。
'''
# 只允许来自 http://localhost:3000 的跨域请求
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
# 初始化 Socket.IO 并配置 WebSocket 的跨域设置
socketio = SocketIO(
    app,  # 绑定到 Flask 应用
    async_mode='eventlet',  # 使用 eventlet 作为异步引擎
    cors_allowed_origins=["http://localhost:3000"],  # 允许 WebSocket 连接的来源
    logger=True,  # 启用 Socket.IO 日志
    engineio_logger=True  # 启用 Engine.IO 底层日志
)

# Define the endpoint to interact with the LangGraph agent
@app.route('/api/agent', methods=['POST'])
def agent():
    data = request.json
    query = data.get("query", "")
    initial_state = {"query": query, "response": ""}
    result = graph.invoke(initial_state)
    return jsonify({"response": result['response']})

# Define a WebSocket event for streaming responses
@socketio.on('connect')
def handle_connect():
    print("Client connected")  # Debugging log

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")  # Debugging log

@socketio.on('stream_query')
async def handle_stream_query(data):
    try:
        query = data.get("query", "")
        initial_state = {"query": query, "response": ""}
        print(f"Received query: {query}")  # Debugging log

        # Stream responses using LangGraph's async stream with stream_mode="values"
        async for chunk in graph.astream(initial_state, stream_mode="values"):
            print(f"Emitting chunk: {chunk['response']}")  # Debugging log
            emit('response_chunk', {'chunk': chunk['response']})  # Stream each chunk to the client

        emit('response_complete', {'message': 'Response streaming complete'})
        print("Streaming complete")  # Debugging log

    except Exception as e:
        print(f"Error in handle_stream_query: {str(e)}")
        emit('error', {'message': str(e)})

# Start the Flask application
if __name__ == '__main__':
    # 只支持http请求
    # app.run(debug=True)

    socketio.run(app, debug=True, host='0.0.0.0', port=5000)