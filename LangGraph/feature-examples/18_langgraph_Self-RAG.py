import asyncio
from typing import List,TypedDict

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
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

def get_prompt_model():
    '''
    2.设置大模型
    :return:
    '''
    prompt = ChatPromptTemplate.from_template("""
    使用以下上下文简洁地回答问题：
    Question: {question} 
    Context: {context} 
    Answer:
    """)
    model = init_chat_model()
    return (prompt | model | StrOutputParser())


# 3. define the graph
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]


# Retrieval Grader setup
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

# Hallucination Grader setup
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the documents, 'yes' or 'no'")

def get_hallucination_grader_model():
    hallucination_prompt = ChatPromptTemplate.from_template("""
    你是一名评分员，评估答案是否基于检索到的文档。
    Documents: {documents} 
    Answer: {generation}
    答案是否基于文件？ 回答 'yes' or 'no'.
    """)
    hallucination_grader = hallucination_prompt | init_chat_model().with_structured_output(
        GradeHallucinations)
    return hallucination_grader

# Answer Grader setup
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

def get_answer_grader_model():
    answer_prompt = ChatPromptTemplate.from_template("""
    您是一名评分员，评估答案是否解决了用户的问题。
    Question: {question} 
    Answer: {generation}
    答案是否回答了这个问题？ 回答 'yes' or 'no'.
    """)
    answer_grader = answer_prompt | init_chat_model().with_structured_output(GradeAnswer)
    return answer_grader

# Define LangGraph functions
def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Grades documents based on relevance to the question.
    Only relevant documents are retained in 'relevant_docs'.
    """
    question = state["question"]
    documents = state["documents"]
    relevant_docs = []

    for doc in documents:
        response = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        # 关键检查：确保响应有效
        if response is None:
            print(f"文档评分返回空: {doc.page_content[:50]}...")
            continue

        if not hasattr(response, "binary_score"):
            print(f"无效响应结构: {response}")
            continue
        if response.binary_score == "yes":
            relevant_docs.append(doc)

    return {"documents": relevant_docs, "question": question}

def decide_to_generate(state):
    """
    Decides whether to proceed with generation or transform the query.
    """
    if not state["documents"]:
        return "transform_query"  # No relevant docs found; rephrase query
    return "generate"  # Relevant docs found; proceed to generate


def grade_generation_v_documents_and_question(state):
    """
    Checks if the generation is grounded in retrieved documents and answers the question.
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Step 1: Check if the generation is grounded in documents
    hallucination_check = hallucination_grader.invoke({"documents": documents, "generation": generation})

    if hallucination_check.binary_score == "no":
        return "not supported"  # Regenerate if generation isn't grounded in documents

    # Step 2: Check if generation addresses the question
    answer_check = answer_grader.invoke({"question": question, "generation": generation})
    return "useful" if answer_check.binary_score == "yes" else "not useful"

def transform_query(state):
    """
    Rephrases the query for improved retrieval if initial attempts do not yield relevant documents.
    """
    transform_prompt = ChatPromptTemplate.from_template("""
    You are a question re-writer that converts an input question to a better version optimized for retrieving relevant documents.
    Original question: {question} 
    Please provide a rephrased question.
    """)

    question_rewriter = transform_prompt | init_chat_model() | StrOutputParser()

    question = state["question"]
    # Rephrase the question using LLM
    transformed_question = question_rewriter.invoke({"question": question})
    return {"question": transformed_question, "documents": state["documents"]}

def get_wordflow_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate,
                                   {"transform_query": "transform_query", "generate": "generate"})
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question,
                                   {"not supported": "generate", "useful": END, "not useful": "transform_query"})

    # Compile the app and run
    app = workflow.compile()

    return app

if __name__ == '__main__':
    load_env()

    # Step 1: 加载文档并预处理
    retriever = load_prepare_docs()

    # Step 2： Set up prompt and model
    rag_chain = get_prompt_model()

    # step3: 获取文档相关性评估模型
    retrieval_grader = get_retrieval_grader_model()

    #step4: 获取幻觉评估模型
    hallucination_grader = get_hallucination_grader_model()

    # step5 :获取答案评估模型
    answer_grader = get_answer_grader_model()

    # step6:定义图
    app = get_wordflow_graph()

    # Display the graph
    # display_graph(app, file_name=os.path.basename(__file__))

    # Example input
    inputs = {"question": "解释不同类型的代理记忆是如何工作的？"}
    for output in app.stream(inputs):
        print(output)






