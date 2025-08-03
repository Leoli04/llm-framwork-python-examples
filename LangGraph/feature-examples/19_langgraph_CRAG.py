import asyncio
from pprint import pprint
from typing import List,TypedDict

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
        collection_name="self-rag-chroma",
        embedding=DashScopeEmbeddings(),
    )
    return vectorstore.as_retriever()



# Step 2: Define Graders and Relevance Model
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader_model():
    retrieval_prompt = ChatPromptTemplate.from_template("""
    您是一名评分员，负责评估文档是否与用户的问题相关:
    Document: {document} 
    Question: {question}
    该文件是否相关？回答 'yes' or 'no'.
    """)
    retrieval_grader = retrieval_prompt | init_chat_model(model_name=AlibabaModel.QWEN_PLUS.get_model_name()).with_structured_output(
        GradeDocuments)
    return retrieval_grader

# Step 3: Query Re-writer
class ImproveQuestion(BaseModel):
    improved_question: str = Field(description="Formulate an improved question.")


def get_query_rewriter_model():
    '''
    模型重写用户问题
    :return:
    '''
    re_write_prompt = ChatPromptTemplate.from_template(
        "Here is the initial question: \n\n {question} \n Formulate an improved question.",
    )
    query_rewriter = re_write_prompt | init_chat_model(model_name=AlibabaModel.QWEN_PLUS.get_model_name()).with_structured_output(
        ImproveQuestion)
    return query_rewriter

# 4.rag模型
def get_rag_chain_model():
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question:
    Question: {question} 
    Context: {context} 
    Answer:
    """)
    rag_chain = prompt | init_chat_model() | StrOutputParser()
    return rag_chain

# 5.1 Define CRAG State
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]

# Step 5.2: Define Workflow Nodes
def retrieve(state):
    '''
    根据用户问题，检索与问题相关的文档
    :param state:
    :return:
    '''
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    '''
    评估文档与问题的是否相关
    :param state:
    :return:
    '''
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search_needed = "No"
    for doc in documents:
        grade = retrieval_grader.invoke({"question": question, "document": doc.page_content}).binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search_needed = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search_needed}

def transform_query(state):
    '''
    使用模型把用户的问题重新描述一遍
    :param state:
    :return:
    '''
    question = state["question"]
    rewritten_question = query_rewriter.invoke({"question": question})
    return {"question": rewritten_question.improved_question, "documents": state["documents"]}


def web_search(state):
    """
    基于模型重新描述的问题，在网上检索
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]
    pprint(question + "\n")
    # Perform web search using TavilySearchResults and extract only the 'content' field for Document
    search_results = TavilySearchResults(k=3).invoke({"query": question})

    # Process results to create Document objects only with page_content
    web_documents = [
        Document(page_content=result["content"]) for result in search_results if "content" in result
    ]

    # Append web search results to the existing documents
    documents.extend(web_documents)

    return {"documents": documents, "question": question}

def generate(state):
    '''
    基于文档回答用户问题
    :param state:
    :return:
    '''
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": generation}

# Step 5.3: Define Decision-Making Logic
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
# 5.4 Build and Compile the Graph
def get_wordflow_graph():
    # Step 6: Build and Compile the Graph
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_searcher", web_search)
    workflow.add_node("generate", generate)

    # Define edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate,
                                   {"transform_query": "transform_query", "generate": "generate"})
    workflow.add_edge("transform_query", "web_searcher")
    workflow.add_edge("web_searcher", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()

    return app



if __name__ == '__main__':
    load_env()

    # Step 1: 加载文档并预处理
    retriever = load_prepare_docs()

    # Step 2： 获取文档相关性评估模型
    retrieval_grader = get_retrieval_grader_model()

    # step3: 重写问题模型
    query_rewriter = get_query_rewriter_model()

    #step4: rag模型
    rag_chain = get_rag_chain_model()

    # step5 :Define CRAG State、Define Workflow Nodes、Build and Compile the Graph
    app = get_wordflow_graph()


    # Display the graph
    display_graph(app, file_name=os.path.basename(__file__))

    # step 6: Example input
    inputs = {"question": "Explain how the different types of agent memory work?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    pprint(value["generation"])






