

## 项目结构

```text
langchain-fastapi=rag-app/
├── app/
│   ├── main.py                  # FastAPI 主入口
│   ├── config.py                # 配置管理
│   ├── dependencies.py          # 依赖注入
│   ├── models.py                # Pydantic 数据模型
│   ├── routers/
│   │   ├── chat.py              # 聊天API路由
│   │   ├── admin.py             # 管理API路由
│   │   └── health.py            # 健康检查路由
│   ├── services/
│   │   ├── vector_store.py      # 向量存储服务
│   │   ├── document_loader.py   # 文档加载处理
│   │   ├── document_manager.py  # 文档管理服务
│   │   ├── retriever.py         # 检索路由服务
│   │   └── llm_service.py       # LLM问答服务
│   ├── core/
│   │   ├── embeddings.py        # 嵌入模型管理
│   │   ├── versioning.py        # 文档版本控制
│   │   └── processing.py        # 文档处理流水线
│   └── utils/
│       ├── logger.py            # 日志配置
│       └── scheduler.py         # 定时任务
├── data/
│   ├── products/                # 产品文档
│   │   ├── pending/            # 待处理文档
│   │   └── processed/          # 已处理文档
│   └── technical/              # 技术文档
│       ├── pending/
│       └── processed/
├── chroma_db/                   # 向量数据库存储
├── tests/                       # 单元测试
├── Dockerfile                   # 容器化配置
├── requirements.txt             # 依赖列表
└── .env                         # 环境变量

```

## 项目启动说明
- 1.环境准备：

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


# 安装依赖
pip install -r requirements.txt

# 创建数据目录
mkdir -p data/products/pending
mkdir -p data/products/processed
mkdir -p data/technical/pending
mkdir -p data/technical/processed
mkdir chroma_db
```

- 2.设置环境变量：

```bash
# 创建 .env 文件
echo "OPENAI_API_KEY=your_api_key" > .env
```

- 3.启动应用：

```bash
python -m app.main
```

- 4.API 使用：

    问答接口：POST /api/v1/chat
    
    文档上传：POST /api/v1/admin/documents/upload
    
    系统状态：GET /api/v1/admin/status

- 5.Docker 运行：

```bash
docker build -t rag-app .
docker run -d -p 8000:8000 -v ./data:/app/data -v ./chroma_db:/app/chroma_db rag-app
```

## 关键功能说明

- 1.文档处理流程：

    - 自动监控 pending 目录处理新文档

    - 支持多种文档格式（PDF、Word、Excel等）

    - 文档处理完成后移至 processed 目录

- 2.向量存储管理：

    - 启动时自动检查文档变更

    - 按需重建向量库

    - 支持多种嵌入模型（OpenAI/SentenceTransformers）

- 3.API 接口：

    - 智能问答接口

    - 文档上传管理

    - 向量库重建接口

    - 系统状态监控

- 4.后台任务：

    - 定时检查向量库完整性

    - 定期扫描文档变更

    - 自动标记需要更新的向量库

- 5.运维支持：

    - 健康检查端点

    - 详细日志记录

    - 容器化部署支持