import datetime
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def load_env():
    from dotenv import load_dotenv
    load_dotenv()  # 自动加载 .env 文件


def init_chat_model(model_name: str):
    from langchain_community.chat_models import ChatTongyi

    return ChatTongyi(model=model_name)

def loader_youtube():
    from langchain_community.document_loaders import YoutubeLoader

    urls = [
        "https://www.youtube.com/watch?v=HAn9vnJy6S4",
        "https://www.youtube.com/watch?v=dA1cHGACXCo",
        "https://www.youtube.com/watch?v=ZcEMLz27sL4",
        "https://www.youtube.com/watch?v=hvAPnpSfSGo",
        "https://www.youtube.com/watch?v=EhlPDL4QrWY",
        "https://www.youtube.com/watch?v=mmBo8nlu2j0",
        "https://www.youtube.com/watch?v=rQdibOsL1ps",
        "https://www.youtube.com/watch?v=28lC4fqukoc",
        "https://www.youtube.com/watch?v=es-9MgxB-uc",
        "https://www.youtube.com/watch?v=wLRHwKuKvOE",
        "https://www.youtube.com/watch?v=ObIltMaRJvY",
        "https://www.youtube.com/watch?v=DjuXACWYkkU",
        "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
    ]

    docs = []
    for url in urls:
        docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())
    # # 给视频添加发布时间
    for doc in docs:
        doc.metadata["publish_year"] = int(
            datetime.datetime.strptime(
                doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
            ).strftime("%Y")
        )

    return docs

def loader_bilibili():
    from langchain_community.document_loaders import BiliBiliLoader

    urls = [
        "https://www.bilibili.com/video/BV13jK2zSE7A/",
        "https://www.bilibili.com/video/BV1Ty4y1Y7B7",
        "https://www.bilibili.com/video/BV1iJutz1Es4",
    ]

    docs = []

    docs.extend(BiliBiliLoader(urls).load())
    for doc in docs:
        # 直接转换时间戳为年份
        pubdate_timestamp = doc.metadata["pubdate"]
        pubdate_dt = datetime.datetime.fromtimestamp(pubdate_timestamp)
        doc.metadata["publish_year"] = pubdate_dt.year

    return docs

def indexing_docs(docs):
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import DashScopeEmbeddings

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    chunked_docs = text_splitter.split_documents(docs)

    tongyi_embeddings = DashScopeEmbeddings(
        model="text-embedding-v4",  # 通义专用嵌入模型
    )

    vectorstore = Chroma.from_documents(
        chunked_docs,
        tongyi_embeddings,
    )

    return vectorstore



class Search(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    query: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    publish_year: Optional[int] = Field(None, description="Year video was published")

if __name__ == '__main__':

    load_env()

    docs = loader_bilibili()

    print([doc.metadata["title"] for doc in docs])
    # # print(docs[0].metadata)
    # print("page_content:",docs[2].page_content)

    # vector_store = indexing_docs(docs)
    # response = vector_store.similarity_search("什么是价值投资")
    #
    # print(response)

    # model = init_chat_model("qwen-plus")
    #
    # system = """You are an expert at converting user questions into database queries. \
    # You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
    # Given a question, return a list of database queries optimized to retrieve the most relevant results.
    #
    # If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system),
    #         ("human", "{question}"),
    #     ]
    # )
    # structured_llm = model.with_structured_output(Search)
    # query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm
    #
    # response = query_analyzer.invoke("2025年什么是价值投资")
    # print(response)

