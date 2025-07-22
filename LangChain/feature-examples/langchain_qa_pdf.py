
def load_env():
    from dotenv import load_dotenv
    load_dotenv()  # 自动加载 .env 文件

def init_chat_model(model_name: str):
    from langchain_community.chat_models import ChatTongyi

    return ChatTongyi(model=model_name)

def load_pdf():
    from langchain_community.document_loaders import PyPDFLoader

    file_path = "./example_data/nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)

    return loader.load()


def index_docs(docs):

    '''

        分割文档，并存储到向量数据库
        :param docs: 要分割存储的文档
        :return: 向量数据库的检索器
    '''

    from langchain_chroma import Chroma
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splitter_docs = text_splitter.split_documents(docs)
    vector_store = Chroma.from_documents(documents=splitter_docs,embedding=DashScopeEmbeddings())
    return vector_store.as_retriever()

def init_rag_chain(model,retriever):
    from langchain.chains.retrieval import create_retrieval_chain
    # from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
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

    question_answer_chain = create_stuff_documents_chain(model, prompt)

    return create_retrieval_chain(retriever, question_answer_chain)


if __name__ == '__main__':

    # 初始化环境变量
    load_env()
    # 加载paf
    docs = load_pdf()

    # print(len(docs))
    # print("docs[1]:",docs[1].page_content[0:100])
    # print(docs[0].metadata)

    # 初始化聊天模型
    # model = init_chat_model("qwen-plus")
    model = init_chat_model("qwen-turbo")

    # 获取向量数据库检索器，并把文档存储到向量数据库
    retriever = index_docs(docs)

    rag_chain = init_rag_chain(model,retriever)

    response = rag_chain.invoke({"input": "What was Nike's revenue in 2023?"})

    print(response)

    print(response["context"][0].page_content)

    print(response["context"][0].metadata)