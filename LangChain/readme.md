## 介绍

关于langchain使用的示例.

## 环境

进入LangChain，在当前目录创建环境
```cmd
# 在当前文件夹创建环境
virtualenv  venv-langchain

# 不用进入当前项目目录，直接指定环境位置
virtualenv 项目位置/LangChain/venv-langchain

#激活环境
venv-langchain\Scripts\activate

# 退出环境
venv-langchain\Scripts\deactivate

```

## 依赖

- jupyter 安装
```cmd
# jupyter
pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 将虚拟环境添加到jupyter内核 中
python -m ipykernel install --name 创建的虚拟环境名字
```

- 通过文件安装
```cmd 

# 安装 requirements.txt文件中的依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


```

- langchain依赖安装
```cmd
pip install langchain==0.3.7
```

- 与第三方模型集成

支持维护的集成：https://python.langchain.com/docs/integrations/providers/

- 通义千问：搜索Alibaba Cloud
- deepseek:  搜索DeepSeek

```
#openai，官方维护
pip install langchain-openai==0.2.3

# huggingface
pip install langchain-huggingface

# deepseek 官方维护
pip install  langchain-deepseek

# 通义千问，通义千问是是社区集成，依赖langchain_community
pip install langchain-community dashscope


```

- 其他依赖
```cmd
# 向量存储
pip install faiss-cpu

pip install  langchain-chroma

```

| 向量数据库       | 是否需要单独安装/部署服务      | 说明                                                         |
 |------------------|-------------------------------|-------------------------------------------------------------|
 | Chroma           | 否                            | 轻量级嵌入式向量库，随应用启动，数据可持久化到本地                        |
 | Milvus           | 是                            | 需要单独部署，类似 MySQL 需要服务运行                         |
 | Pinecone         | 否（但需要注册云服务）         | 通过 API 访问，无需自部署，但需要网络连接和 API key            |
| faiss        | 否                         | 高性能本地检索引擎            |

-  查看自己安装库的版本
```shell
#方式一
pip show langchain

#方式二
# Windows 系统
pip list | findstr /i "langchain"

# Linux/macOS 系统
pip list | grep -i "langchain"
```


## 密钥

 有三种种方式配置：
 - 通过环境变量的方式配置(需要重启编辑器才能获取环境变量)
   参考：https://help.aliyun.com/zh/model-studio/first-api-call-to-qwen#44cd5f8592qor
 - 通过python内置函数`getpass()`安全输入的方式 
   
    ```python
      DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or getpass("请输入密钥：")
      os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    ```
- 通过.env文件配置

   ```python
  
   from dotenv import load_dotenv
   load_dotenv()
  
   ```
  
## 数据库数据

### 数据初始化

执行下面脚本创建Chinook.db数据库并初始化数据，具体参考[SQLite 示例数据库](https://database.guide/2-sample-databases-sqlite/)
```shell
# 启动 SQLite 命令行工具并创建/连接数据库文件
sqlite3 Chinook.db

#执行指定的 SQL 脚本文件来初始化数据库
.read Chinook_Sqlite.sql

# 退出sqlite命令
.exit
#或者
.quit
```

退出sqlite命令行 快捷键：

Linux/Mac：按 Ctrl + D

Windows：按 Ctrl + Z 然后按 Enter


### 连接数据库

目录结构如下：
```text
项目/
└── langchain/
    ├── feature-examples/
    │   └── your_script.py          # 您的 Python 文件位置
    │       
    └── Chinook.db           # 数据库文件位置
```

方式一：使用os
```python
    # 获取当前 Python 文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建相对路径：从 python/ 目录向上退两级到 langchain/
    db_relative_path = os.path.join(current_dir, '..', 'Chinook.db')

    # 转换为绝对路径并规范化
    db_absolute_path = os.path.abspath(db_relative_path)

    print("db_absolute_path:",db_absolute_path)
    # 创建数据库连接
    uri = f"sqlite:///{db_absolute_path}"
```

方式二:使用pathlib
```python
    from pathlib import Path
    # 使用 pathlib 处理路径
    current_dir = Path(__file__).resolve().parent
    db_path = current_dir.parent / "Chinook.db"
    # 创建数据库连接
    # 创建 URI (Windows 需要特殊处理)
    if os.name == 'nt':  # Windows 系统
        uri = f"sqlite:///{db_path.resolve()}"
    else:  # Linux/Mac
        uri = f"sqlite:////{db_path.resolve()}"
```



  
## 资料
- [langchain手册](https://github.com/langchain-ai/langchain/blob/master/cookbook/README.md)
- [langchain官网](https://python.langchain.com/docs/introduction/)
- [langchain中文教程](https://www.langchain.com.cn/docs/tutorials/)

