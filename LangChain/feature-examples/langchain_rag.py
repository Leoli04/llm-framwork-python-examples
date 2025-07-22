import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()  # 自动加载 .env 文件

    '''
    bs4指的是Beautiful Soup库，它是一个用于解析HTML和XML文档的Python库
    '''
    # 创建一个文档解析过滤器,只保留文档的class属性为"post-title", "post-header", "post-content"
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer}
    )

    # 1.索引-加载文档
    docs = loader.load()
    # print(len(docs[0].page_content))
    # print(docs[0].page_content[:500])

    #  2.索引-分割
    # 把文档拆分成每块1000个字符 并在块之间重叠200个字符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # print(len(all_splits))
    # print(all_splits[10].metadata)

    #     3.索引-存储
    from langchain_chroma import Chroma
    from langchain_community.embeddings import DashScopeEmbeddings

    # 默认数据存储在内存
    vector_store = Chroma.from_documents(documents=all_splits, embedding=DashScopeEmbeddings())

    # 4.检索
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

    # 5.生成
    from langchain_community.chat_models.tongyi import ChatTongyi
    from langchain import hub

    model = ChatTongyi(
        model="qwen-plus",  # 明确指定模型
        max_retries=3  # 失败重试

    )
    prompt = hub.pull("rlm/rag-prompt")

    # example_messages = prompt.invoke(
    #     {"context": "filler context", "question": "filler question"}
    # ).to_messages()
    #
    # print(example_messages[0].content)

    from langchain_core.output_parsers import  StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    print("正在向通义千问发送请求...")
    response = rag_chain.invoke("What is Task Decomposition?")
    print("\n=== 模型响应 ===")
    print(response)








