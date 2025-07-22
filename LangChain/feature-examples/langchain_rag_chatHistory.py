

def chat_without_history(model,retriever):
    # 模型提示词
    from langchain_core.prompts import ChatPromptTemplate
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("正在向通义千问发送请求...")
    response = rag_chain.invoke({"input": "What is Task Decomposition?"})
    print("\n=== 模型响应 ===")
    print(response["answer"])



def chat_with_history(model,retriever):
    '''
    支持历史消息
    (查询, 聊天历史) -> 大型语言模型 -> 重新表述的查询 -> 检索器
    '''
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("正在向通义千问发送请求...")
    response = conversational_rag_chain.invoke(
        {"input": "What is Task Decomposition?"},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )
    print("\n=== 模型响应 ===")
    print(response["answer"])

    print("正在向通义千问发送请求...")
    response = conversational_rag_chain.invoke(
        {"input": "What are common ways of doing it?"},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )
    print("\n=== 模型响应 ===")
    print(response["answer"])


def chat_with_agent_and_tool(model,retriever):
    '''
    使用代理简化chat_with_history流程

    代理利用大型语言模型的推理能力在执行过程中做出决策。使用代理可以将一些对检索过程的自由裁量权转移出去。
    '''
    from langchain.tools.retriever import create_retriever_tool
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage
    ### 构建检索器工具 ###
    tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
    )
    tools = [tool]
    memory = MemorySaver()

    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # 模拟第一次对话
    config = {"configurable": {"thread_id": "abc123"}}
    print("模拟第一次对话:")
    for s in agent_executor.stream(
            {"messages": [HumanMessage(content="Hi! I'm bob")]}, config=config
    ):
        print(s)
        print("----")

    print("模拟第二次对话:")
    query = "What is Task Decomposition?"

    for s in agent_executor.stream(
            {"messages": [HumanMessage(content=query)]}, config=config
    ):
        print(s)
        print("----")

    # 模拟第三次对话
    print("模拟第三次次对话:")
    query = "What according to the blog post are common ways of doing it? redo the search"

    for s in agent_executor.stream(
            {"messages": [HumanMessage(content=query)]}, config=config
    ):
        print(s)
        print("----")





if __name__ == '__main__':

    '''
    1.加载环境变量
    '''
    from dotenv import load_dotenv

    load_dotenv()  # 自动加载 .env 文件

    '''
    2.创建聊天模型实例
    '''
    from langchain_community.chat_models.tongyi import ChatTongyi

    model = ChatTongyi(model="qwen-plus")

    '''
    3.索引外部文档
    '''
    # 加载文档
    import bs4
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs = loader.load()

    #  分割文档
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 索引并存储

    from langchain_chroma import Chroma
    from langchain_community.embeddings import DashScopeEmbeddings
    # 基于分割的文档和潜入模型创建向量数据库
    vector_store = Chroma.from_documents(documents=splits,embedding=DashScopeEmbeddings(model="text-embedding-v1",))

    '''
    4.检索文档并生成答案
    '''
    # 基于向量数据库创建检索器
    retriever = vector_store.as_retriever()

    # chat_without_history(model,retriever)

    # chat_with_history(model,retriever)
    chat_with_agent_and_tool(model,retriever)










