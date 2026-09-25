# 智导（CommercePilot）

`智导（CommercePilot）` 是一个把电商推荐与客服 RAG 知识检索能力融合到一起的智能导购客服系统：

- 一边保留 `Salesperson` 的推荐、上下文理解、对话表达和记忆能力。
- 一边复用原有 RAG 的文档解析、三级分块、混合检索、rerank、step-back / HyDE、Auto-merging 等技术组件。
- 最终让商品推荐、客服问答、售后政策、FAQ、导购话术和回复质检走进同一条交互链路。

## 项目结构

- `backend/`：后端主目录，包含推荐 Agent、客服领域模块、RAG、数据库和接口层。
- `backend/rag/`：RAG 能力层，负责检索、分块、入库、重写、存储和工具方法。
- `backend/customer_service/`：客服领域能力层，共享执行器按任务加载咨询、FAQ、话术和质检模板。
- `frontend/`：Vue 工作台，面向用户提供导购对话和资料库页面。
- `config/agents.yaml`：统一管理 Agent、Broker 和 Mesh 静态参数。
- `data/`：知识库、索引和上传文件的本地目录。
- `docs/`：项目文档。
- `scripts/`：脚本与辅助工具。

## 核心方向

- 统一入口：后端最终希望从同一个 API 服务里同时提供推荐和 RAG 能力。
- 分层复用：推荐和检索不要互相污染，通用能力尽量放在 `rag/`、`services/`、`repositories/` 这种基础层。
- 前后端分域：聊天、检索 trace、资料库管理分开组织，便于后续扩展。

## Adaptive Agent Mesh Demo

导购推荐默认使用一个单进程的动态 Agent Mesh Demo：

- LLM Planner 根据请求生成结构化 Task DAG；能力、依赖、循环、终止节点和
  任务规模校验失败时，自动使用确定性规则计划。
- 同一能力默认只注册一个 Agent；Hybrid Broker 在只有一个候选时直接路由，
  不调用 LLM。保留多候选路由接口，供后续确有不同实现时扩展。
- 商品召回统一在一个入口内按场景选择语义、画像或热门策略，保留混合召回能力。
- 知识检索统一在一个入口内选择快速或标准模式；候选评估由内部服务
  选择轻量规则或语义模式。沿用复杂度判定，复杂请求和反馈修订使用语义评估。
- 知识检索等待商品召回与库存过滤完成，再为多个候选商品并行检索证据。
- 各任务通过请求级 Blackboard 交换结果。
- 最终回复由统一 Quality Reviewer 审核。可修正时，Replanner 只追加“重写回复→复审”两步；
  模型或计划校验失败时使用规则计划。两种编排均禁止审核后再生成未审核的正文。
- 最终 Judge 结果会回流到参与推荐质量的 Agent，后续 Broker 使用业务成功率
  进行选择；请求级和任务级评测事件会持久化到 SQL 数据库。
- Mesh 异常时自动回退原有 Supervisor Workflow。

前端导购页会展示执行波次、Agent、内部策略、Broker 分数和 Replan 状态。
编排模式、最大重规划次数、LLM Planner/Broker/Replanner 开关和 Broker 权重统一在
`config/agents.yaml` 中维护。

## Agent 配置

统一配置从最初 27 项经两轮精简为 10 项：

| Agent | 职责 |
| --- | --- |
| conversation-understanding | 导购需求解析、客服任务路由与拆分 |
| image-understanding | 图片理解 |
| memory-update | 长期记忆提取与冲突处理 |
| product-recommendation | 商品召回，内部执行库存和硬约束过滤 |
| response-generation | 导购、闲聊、客服聊天以及结构化 FAQ/话术生成 |
| quality-reviewer | 最终回复审核、修订后复审、用户主动请求的回复质检 |
| knowledge-retrieval | 快速或标准知识检索 |
| llm-planner | 推荐任务计划 |
| llm-broker | 能力路由；单候选不调用模型 |
| llm-replanner | 失败后的回复修订计划 |

候选评估已移为 `services/candidate_evaluation.py`，仅在复杂需求或修订时使用语义模型；
参数位于 `services.candidate-evaluation`，不注册 Agent Card。
`services/product_constraints.py` 集中处理库存、预算和类目规则，不调用 LLM。
普通确认、问候等固定文本仍直接回复；生成式回复经过统一审核。

推荐主链路为“召回并过滤→按需取证→内部评估并生成→最终审核”。
审核收到实际最终正文、商品短文案、商品事实和知识库原文；至多修订一次。
审核失败或不可用时输出保守提示，清空推荐展示，不退回未经审核的原候选。

客服聊天共用回复生成器的执行图，`customer_service/task_profiles.py` 保留四类任务的
提示词和工具白名单。质检模式使用同一 Quality Reviewer 模型；路由共用需求理解角色。
FAQ/话术等结构化 API 同样交付前审核，未通过的草稿不保存、不返回。
客服流式接口继续推送检索进度，正文在审核通过后才发送，不提前输出模型草稿。

历史评测不改写原始记录，旧角色按新 ID 汇总；已移除的评估、库存 Agent 历史数据
单列为 `historical_internal_steps`。旧候选审核与新最终回复审核的统计口径不同，
不能直接据此判断质量提升。
自定义 `ECOM_AGENT_CONFIG_FILE` 需要同步最新 Agent ID、executor、capabilities、
model_profiles 和 services 配置。

Agent 配置按职责分为三层：

- `.env`：模型密钥、模型地址、数据库、Redis 和 Milvus 等环境配置。
- `config/agents.yaml`：模型参数、超时、重试、capability、Agent Card、
  Broker 权重和 Mesh 编排参数。
- Registry 运行状态：调用次数、成功率、实时负载和观测延迟；负载仅用于
  运维观测，不再参与 Broker 评分。

启动时 YAML 会经过 Pydantic 校验；无效模型 Profile、Broker 权重错误、
未声明 capability 或不在代码白名单中的 Mesh executor 会直接终止启动。
可使用 `ECOM_AGENT_CONFIG_FILE` 指定其他配置文件。

管理员可通过 `GET /api/v1/agent-evaluations?days=30` 查看业务成功率、Judge
通过率、Replan 修复率、Agent/Capability 成功率、延迟和场景报价校准误差。

## 四类模型配置

模型统一按四类用途配置，由代码选择调用入口：

| 用途 | 配置 | 使用位置 |
| --- | --- | --- |
| 文本 | `TEXT_LLM`、`TEXT_API_KEY`、`TEXT_BASE_URL` | 商品翻译、聊天、规划、记忆、审核、客服及 RAG 查询改写 |
| 视觉 | `VISION_LLM`、`VISION_API_KEY`、`VISION_BASE_URL` | 图片理解 Agent |
| 向量 | `EMBEDDING_MODEL`、`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_DIMENSION` | 商品、用户画像、客服文档及查询向量 |
| 重排序 | `RERANK_MODEL`、`RERANK_MODEL_PATH` | 标准 RAG 的文档相关性重排序 |

模型名称、密钥和地址在 `.env` 管理，统一聊天入口是
`backend/core/model_clients.py`；`config/agents.yaml` 只调整每个 Agent 的
温度、输出长度和运行策略。各 Agent 不再单独决定模型名称。
文本角色配置为 `qwen3.8-max`；它也具有视觉能力，但本项目通过独立角色选择用途。
视觉仍为 `qwen-vl-max`；向量统一为 `text-embedding-v4`（1024 维）。
配置正确不代表账户有可用额度，尤其旧视觉模型的免费额度可能已过期。

BGE 使用 `BAAI/bge-reranker-v2-m3`。请自行下载完整目录（权重、config、
tokenizer 等文件），再填写项目根目录相对路径，例如
`RERANK_MODEL_PATH=model/bge-reranker-v2-m3`。
代码设置 `local_files_only=True`，不会自动下载或执行模型仓库的自定义代码。
默认使用 CPU，可通过 `RERANK_DEVICE` 调整。空路径且未配置远程 API 时不启用；
本地目录缺失、模型不兼容或推理失败时保留原召回顺序，并写入 `rerank_error`。
快速 RAG 路径仍有意跳过重排序。商品候选评估仍由文本/规则服务完成，
不会自动改成 BGE 排序。

四类模型的名称、API Key 和 URL 分开配置，不再读取 `ECOM_LLM_*`、
`ARK_API_KEY`、`MODEL`、`GRADE_MODEL` 等旧模型变量，也不通过环境变量引用复用连接信息。
本地 BGE Reranker 不经过 API，因此本地模式下 `RERANK_API_KEY` 和
`RERANK_BASE_URL` 可以留空。

**向量迁移：** 商品、用户和客服文档集合名会自动追加 `_e<模型配置指纹>`，
BM25 状态文件也按同一指纹隔离。首次升级或更换 Embedding 的模型、地址、维度后，
需要重新执行商品导入/管理员向量构建、重建用户向量，并重新上传或导入客服文档。
旧集合、旧 BM25 文件及原始文档不会自动删除；新查询不会混用旧模型向量。
仅改文本模型或 Reranker 不需要重新生成向量。

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

关闭 PostgreSQL、Redis 和 Milvus 容器：

```bash
docker compose down
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
