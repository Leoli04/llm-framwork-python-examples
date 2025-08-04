## 介绍

关于langgraph使用的示例

## 涉及的第三方工具

- TextBlob适合用于快速进行基本的文本分析和处理任务，如情感分析、拼写检查和翻译等。安装`pip install textblob`

- Tavily是一个搜索API，用于从网络上获取实时搜索结果。安装`pip install tavily-python`
  
- psutil 是一个跨平台的 Python 库（Process and System Utilities），用于监控系统资源和运行进程。安装`pip install psutil  # 基础安装
pip install psutil[all]  # 包含额外功能（如进程线程操作）`
  
- PyPDF 是 PyPDF2 库的继任者（PyPDF2 已不再积极维护）。它是一个纯 Python 库，用于处理 PDF 文件，支持读取、拆分、合并、转换等操作。安装`pip install pypdf`

- Unstructured 是一个更强大的文档处理库，专为处理非结构化数据（如 PDF、Word、HTML 等）设计。它能保留文档结构（标题、列表等），适合复杂文档的预处理。安装`pip install unstructured[pdf]`

- Beautiful Soup 是一个用于解析 HTML 和 XML 文档的流行 Python 库，常用于网页抓取和数据提取。安装：`pip install beautifulsoup4`

- PythonREPLTool 是 LangChain 框架中的一个工具类，它允许语言模型（LLM）通过执行 Python 代码来解决问题。这个工具提供了一个安全的 Python REPL（Read-Eval-Print Loop，交互式解释器）环境，让语言模型能够执行代码并获取结果。
安装`pip install langchain langchain-experimental`
  
- pytest是一个测试工具(比unittest更先进),安装`pip install pytest`
  
## 向量数据库

免费

- FAISS——由Facebook人工智能团队开发的高性能、开源库，适用于本地部署。
  
- Chroma——轻量级，作为库在本地运行，并与Python集成良好。
    
- Lance——开源向量存储库，优化用于高效存储和检索。

设备端存储：

- Milvus——开源，可以部署在本地或云上；支持动态索引。
  
- Weaviate——在本地和云上运行，具有强大的查询语言和知识图谱支持。

基于云的向量存储：

- Pinecone——基于云，提供可扩展和托管的向量存储。
  
- Google Vertex Matching Engine——由谷歌提供的托管服务，用于快速和可扩展的相似性搜索。
  
- AWS Kendra——与AWS集成的完全托管的搜索服务。
  
- Redis Vector Search——支持向量相似性搜索的Redis模块，优化用于云环境。
  
- Vespa——开源的搜索和推荐引擎，适用于本地部署。