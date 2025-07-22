from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

def test_vector_similarity_search(vector_store):

    print("正在向通义千问发送请求...")
    response = vector_store.similarity_search("cat")
    print("\n=== similarity_search模型响应 ===")
    print(response)

def test_vector_similarity_search_with_score(vector_store):

    print("正在向通义千问发送请求...")
    response = vector_store.similarity_search_with_score("cat")
    print("\n=== similarity_search_with_score模型响应 ===")
    print(response)

def test_vector_similarity_search_by_vector(vector_store,text):
    print("正在向通义千问发送请求...")
    query_embedding = tongyi_embeddings.embed_query(text)
    response = vector_store.similarity_search_by_vector(query_embedding)
    print("\n=== similarity_search_by_vector模型响应 ===")
    print(response)

def test_vector_rag(vector_store):
    from langchain_community.chat_models.tongyi import ChatTongyi
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    # 定义检索器
    # retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

    # 检索器：返回第一个，和上面自定义的方式等价
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    model = ChatTongyi(
        model="qwen-plus",  # 明确指定模型
        max_retries=3  # 失败重试

    )

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

    print("正在向通义千问发送请求...")
    response = rag_chain.invoke("tell me about cats")
    print("\n=== test_vector_rag 模型响应 ===")
    print(response)


if __name__ == '__main__':
    '''
    使用langchain_chroma作为向量数据库，使用前需要安装
    pip install langchain langchain-chroma
    '''

    # 1.初始化通义嵌入模型
    tongyi_embeddings = DashScopeEmbeddings(
        model="text-embedding-v4",  # 通义专用嵌入模型
    )
    # 2.把文档添加到向量存储中
    vector_store = Chroma.from_documents(
        documents,
        embedding=tongyi_embeddings,
    )

    # 3. 执行查询（添加异常处理）
    try:

        # test_vector_similarity_search(vector_store)

        # test_vector_similarity_search_with_score(vector_store)

        # test_vector_similarity_search_by_vector(vector_store,"cat")

        test_vector_rag(vector_store)




    except Exception as e:
        print(f"\n⚠️ 请求失败: {str(e)}")
        print("可能原因:")
        print("- 无效的API密钥")
        print("- 未开通通义千问服务（需在阿里云控制台开通）")
        print("- 网络连接问题")
        print(f"- 账户余额不足（请访问: https://dashscope.console.aliyun.com/）")