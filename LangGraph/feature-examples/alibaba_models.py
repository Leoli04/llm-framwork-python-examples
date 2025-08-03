from enum import Enum


class AlibabaModel(Enum):
    """
    阿里云通义千问系列模型枚举
    包含官方提供的常用模型名称
    """
    # 通义千问主系列模型
    QWEN_TURBO = "qwen-turbo"  # 超高速版本，适用于轻量级任务
    QWEN_PLUS = "qwen-plus"  # 增强版，平衡性能与速度
    QWEN_MAX = "qwen-max"  # 旗舰版，最高性能
    QWEN_MAX_LONGCONTEXT = "qwen-max-longcontext"  # 支持128K长文本上下文

    # 通义千问开源模型
    QWEN_1_5_0_5B = "qwen1.5-0.5b"  # 0.5B参数小型模型
    QWEN_1_5_1_8B = "qwen1.5-1.8b"  # 1.8B参数基础模型
    QWEN_1_5_7B = "qwen1.5-7b"  # 7B参数标准模型
    QWEN_1_5_14B = "qwen1.5-14b"  # 14B参数大模型
    QWEN_1_5_72B = "qwen1.5-72b"  # 72B参数超大模型

    # 通义千问多模态模型
    QWEN_VL_PLUS = "qwen-vl-plus"  # 视觉语言增强模型
    QWEN_VL_MAX = "qwen-vl-max"  # 视觉语言旗舰模型

    # 通义千问音频模型
    QWEN_AUDIO = "qwen-audio"  # 音频理解模型

    # 代码专用模型
    QWEN_CODE = "qwen-code"  # 代码生成与理解专用

    # 金融专用模型
    QWEN_FINANCE = "qwen-finance"  # 金融领域优化模型

    def get_model_name(self) -> str:
        """获取模型名称字符串"""
        return self.value

    @classmethod
    def list_models(cls) -> list:
        """列出所有可用模型名称"""
        return [model.value for model in cls]

    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        """检查是否为有效的模型名称"""
        return model_name in cls.list_models()

    @classmethod
    def get_default_model(cls) -> str:
        """获取推荐的默认模型"""
        return cls.QWEN_TURBO.value


# 使用示例
if __name__ == "__main__":
    # 获取所有模型列表
    print("可用阿里云模型:")
    for model in AlibabaModel.list_models():
        print(f"- {model}")

    # 检查模型有效性
    test_model = "qwen-max"
    print(f"\n'{test_model}' 是否有效? {AlibabaModel.is_valid_model(test_model)}")

    # 使用推荐模型
    print(f"\n推荐默认模型: {AlibabaModel.get_default_model()}")

    # 在代码中使用枚举
    selected_model = AlibabaModel.QWEN_TURBO
    print(f"\n选择的模型: {selected_model.get_model_name()}")