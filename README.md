# 智导（CommercePilot）

`智导（CommercePilot）` 是一个把电商推荐与客服 RAG 知识检索能力融合到一起的智能导购客服系统：

- 一边保留 `Salesperson` 的推荐、上下文理解、对话表达和记忆能力。
- 一边复用原有 RAG 的文档解析、三级分块、混合检索、rerank、step-back / HyDE、Auto-merging 等技术组件。
- 最终让商品推荐、客服问答、售后政策、FAQ、导购话术和回复质检走进同一条交互链路。

## 项目结构

- `backend/`：后端主目录，包含推荐 Agent、客服领域模块、RAG、数据库和接口层。
- `backend/rag/`：RAG 能力层，负责检索、分块、入库、重写、存储和工具方法。
- `backend/customer_service/`：客服领域能力层，包含 FAQ、导购话术、回复质检和客服路由 Agent。
- `frontend/`：Vue 工作台，面向用户提供导购对话和资料库页面。
- `data/`：知识库、索引和上传文件的本地目录。
- `docs/`：项目文档。
- `scripts/`：脚本与辅助工具。

## 核心方向

- 统一入口：后端最终希望从同一个 API 服务里同时提供推荐和 RAG 能力。
- 分层复用：推荐和检索不要互相污染，通用能力尽量放在 `rag/`、`services/`、`repositories/` 这种基础层。
- 前后端分域：聊天、检索 trace、资料库管理分开组织，便于后续扩展。

## 当前定位

仓库已经形成代码层面的融合型 MVP：销售推荐会按需读取 RAG
证据并返回引用，导购对话和资料库页面均已接入对应 API，首次启动还可选择写入
幂等的演示商品。当前剩余工作主要是安装依赖、填写模型配置、启动外部服务并
完成真实环境端到端联调。

## 本地开发

后端要求 Python 3.11–3.13，前端要求 Node.js 和 pnpm。先准备配置：

```bash
cp .env.example .env
# 至少填写模型密钥、32 位以上 JWT_SECRET_KEY 和 ECOM_ADMIN_PASSWORD
```

启动 PostgreSQL、Redis 和 Milvus：

```bash
docker compose up -d
docker compose ps
```

本地开发使用 Milvus Standalone 的内嵌 etcd 与本地持久化存储，数据保存在
Docker 的 `milvus-data` 卷中，因此不需要额外启动 etcd 和 MinIO 容器。

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirement.txt
cd frontend
pnpm install
cd ..
```

同时启动前后端：

```bash
./scripts/dev.sh
```

也可以分别在 `backend/` 运行
`uvicorn main:app --reload --port 8000`，在 `frontend/` 运行
`pnpm dev`。前端开发服务器会把 `/api` 和 `/health` 代理到
`127.0.0.1:8000`。

启动后可访问：

- 前端：`http://127.0.0.1:5173`
- OpenAPI：`http://127.0.0.1:8000/docs`
- 存活检查：`http://127.0.0.1:8000/health`
- 完整就绪检查：`http://127.0.0.1:8000/health/ready`

`.env.example` 默认开启 `ECOM_SEED_DEMO_DATA=true`，仅在商品表为空时写入
演示商品、库存和行为数据，不会覆盖已有目录。商品向量可在管理员页面中重建。
