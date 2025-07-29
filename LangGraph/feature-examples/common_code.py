from langchain_core.language_models import BaseChatModel


def load_env():
    from dotenv import load_dotenv
    load_dotenv()  # 自动加载 .env 文件


def init_chat_model(model_name: str = "qwen-turbo") -> BaseChatModel:
    from langchain_community.chat_models import ChatTongyi

    return ChatTongyi(model=model_name)
