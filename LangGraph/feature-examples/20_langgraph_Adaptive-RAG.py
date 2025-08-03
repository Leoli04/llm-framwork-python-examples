import asyncio
from pprint import pprint
from typing import List, TypedDict, Literal

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field

from common_code import load_env, init_chat_model
from display_graph import display_graph
from alibaba_models  import AlibabaModel

import os





def load_prepare_docs():
    '''
     #1. Index 3 websites by adding them to a vector DB
    :return:
    '''
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="adaptive-rag",
        embedding=DashScopeEmbeddings(),
    )
    return vectorstore.as_retriever()

# Step 2: Define routing model
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"]

def get_question_router_model():
    '''
    问题路由模型
    :return:
    '''
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at routing a user question to vectorstore or web search."),
        ("human", "{question}")
    ])

    question_router = route_prompt | init_chat_model().with_structured_output(RouteQuery)
    return question_router


# Step 3: Define Graders and Relevance Model
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader_model():
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate if the document is relevant to the question. Answer 'yes' or 'no'."),
        ("human", "Document: {document}\nQuestion: {question}")
    ])
    retrieval_grader = grade_prompt | init_chat_model(model_name=AlibabaModel.QWEN_PLUS.get_model_name()).with_structured_output(
        GradeDocuments)
    return retrieval_grader

# 网络搜索
def web_search(state):
    web_search_tool = TavilySearchResults(k=3)
    search_results = web_search_tool.invoke({"query": state["question"]})
    web_documents = [Document(page_content=result["content"]) for result in search_results if "content" in result]
    return {"documents": web_documents, "question": state["question"]}

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Define nodes for query handling
def retrieve(state):
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}


def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search_needed = "No"

    for doc in documents:
        grade = retrieval_grader.invoke({"question": question, "document": doc.page_content}).binary_score
        if grade == "yes":
            filtered_docs.append(doc)
        else:
            web_search_needed = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search_needed}

def generate(state):
    prompt_template = """
    Use the following context to answer the question concisely and accurately:
    Question: {question} 
    Context: {context} 
    Answer:
    """

    # Define ChatPromptTemplate using the above template
    rag_prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create the RAG generation chain with LLM and output parsing
    rag_chain = (
        rag_prompt |
        init_chat_model() |
        StrOutputParser()
    )
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": generation}

# Route question based on source
def route_question(state):
    source = question_router.invoke({"question": state["question"]}).datasource
    return "web_search" if source == "web_search" else "retrieve"

# 5.4 Build and Compile the Graph
def get_wordflow_graph():
    # Compile and Run the Graph
    workflow = StateGraph(GraphState)
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.add_conditional_edges(START, route_question, {"web_search": "web_search", "retrieve": "retrieve"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()

    return app



if __name__ == '__main__':

    load_env()

    # Step 1: 获取文档检索器：加载文档并预处理,
    retriever = load_prepare_docs()

    # Step 2： 问题路由模型
    question_router = get_question_router_model()

    # step3: 文档与问题的关联性评估模型
    retrieval_grader = get_retrieval_grader_model()


    # step5 :Define CRAG State、Define Workflow Nodes、Build and Compile the Graph
    app = get_wordflow_graph()


    # Display the graph
    display_graph(app, file_name=os.path.basename(__file__))

    # step 6: Example input
    # Run with example inputs
    inputs = {"question": "What are the types of agent memory?"}
    for output in app.stream(inputs):
        print(output)






