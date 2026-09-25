# CommercePilot Agent Mesh 开发计划

> 最后更新：2026-09-09
>
> 当前阶段：Adaptive Agent Mesh 增强 Demo（LLM Planner + Hybrid Broker）
>
> 默认模式：`mesh`

## 1. 当前结论

项目已经从固定的 Supervisor Workflow，升级为一个可运行的 Adaptive Agent Mesh Demo。

当前主链路具备：

```text
请求路由
  → LLM Planner 生成并校验 Task DAG
  → Hybrid Broker 结合 LLM 场景评估与运行指标选择 Agent
  → Executor 按依赖并行执行
  → Blackboard 共享状态
  → Judge 审核结果
  → Replanner 追加修正任务
  → Response Synthesizer 生成回答
  → 异常时回退 Supervisor Workflow
```

需要注意：当前架构形态已经完成，但 Planner、Broker、Replanner、Blackboard 和容错机制仍有部分 Demo 化实现，尚未达到生产级。

## 2. 当前完整请求链路

### 2.1 普通文本请求

```text
POST /api/v1/chat
  → 加载会话历史
  → ChatRouterAgent
      ├─ 普通对话
      │   → ConversationReplyAgent
      │   → 更新会话记忆
      │   → 返回回答
      └─ 商品推荐
          → ConversationContextAgent
          → 加载长期用户偏好
          → 构造 RecommendationRequest
          → AdaptiveAgentMeshOrchestrator
          → 生成回答、商品、引用和执行轨迹
          → 更新会话记忆
          → 返回前端
```

### 2.2 图片请求

```text
图片
  → ImageUnderstandingAgent
  → ConversationContextAgent
  → AdaptiveAgentMeshOrchestrator
  → 返回推荐结果
```

### 2.3 Mesh 内部链路

```text
LLM Planner
  → Pydantic + 初始计划安全校验
  → 失败时使用规则 Planner
  → Initial Task DAG
      Wave 1: product_recall
      Wave 2: inventory_filter
      Wave 3: knowledge_retrieval（RAG 开启时，候选商品内部并行检索）
      Wave 4: candidate_evaluation
      Wave 5: recommendation_judge
  → Judge 通过
      Wave 6: response_synthesis
  → Judge 不通过且建议重试
      LLM Replanner 输出结构化 Task DAG
      → Pydantic + 安全规则校验
      → 按 Judge 问题动态追加任务
      → 校验失败时使用规则 Replanner
```

## 3. 已经实现的能力

### 3.1 Mesh 模式接入

- [x] 默认编排模式设置为 `mesh`。
- [x] 保留原 Supervisor Workflow。
- [x] 支持通过配置在 Mesh 和 Workflow 之间切换。
- [x] Mesh 执行异常时自动回退 Supervisor Workflow。
- [x] 文本推荐、图片推荐和直接推荐接口统一接入新的 orchestrator。

Agent 配置入口：

```env
ECOM_AGENT_CONFIG_FILE=config/agents.yaml
```

编排模式、最大 Replan 次数、LLM Planner/Broker/Replanner 开关和最大动态任务数已经迁移到
`config/agents.yaml`。

### 3.2 Planner 与 Task DAG

- [x] 定义 `MeshTask`。
- [x] 定义 `MeshExecutionPlan`。
- [x] 支持任务 ID、能力声明、依赖关系和输入参数。
- [x] 根据是否启用 RAG，动态决定是否加入知识检索任务。
- [x] 生成初始推荐 DAG。
- [x] Judge 完成后生成后续执行计划。
- [x] 校验重复任务 ID。
- [x] 校验不存在的任务依赖。
- [x] 检测没有可执行节点的异常 DAG。
- [x] LLM Replanner 使用 Pydantic 结构化输出生成后续 DAG。
- [x] 动态计划执行 capability 白名单、注册表、依赖、循环和规模检查。
- [x] 结构化输出或安全检查失败时回退规则计划。
- [x] LLM Planner 使用 `InitialPlanDecision` 结构化输出生成初始 DAG。
- [x] 初始计划执行能力、依赖、循环、执行顺序和唯一终止节点检查。

当前限制：

- 动态计划尚未执行 Token 预算和最大深度检查。

### 3.3 Broker 与 Agent Registry

- [x] 定义 Agent Card。
- [x] Agent 可以声明一个或多个 capability。
- [x] Agent Registry 支持注册和能力查询。
- [x] Broker 可以从相同能力的多个候选 Agent 中选择最高分 Agent。
- [x] `product_recall` 注册语义、画像和热度三个真实召回 Agent。
- [x] `knowledge_retrieval` 注册快速混合检索与标准检索两个 Agent。
- [x] `candidate_evaluation` 注册确定性轻量评估与语义评估两个 Agent。
- [x] Broker 评分收敛为历史业务成功率、实际延迟、配置成本和场景匹配度。
- [x] LLM 对所有候选 Agent 输出结构化场景评估，并与规则报价加权融合。
- [x] 候选列表不完整、输出非法或模型不可用时自动退回指标 Broker。
- [x] 删除不易量化的任务置信度、负载、独立 Token 成本评分。
- [x] 场景匹配度由候选 Agent 的确定性 `bid()` 动态返回，不再由 Broker
  读取静态 `routing_hints`。
- [x] Agent Bid 返回 `can_handle`、`scenario_score` 和可解释原因。
- [x] Broker 选择结果和分数写入执行轨迹。
- [x] 执行结果自动更新 Agent 成功率和指数移动平均延迟。
- [x] 历史成功率使用配置先验和实际调用记录进行平滑，避免小样本剧烈波动。
- [x] Trace 记录全部候选 Agent、评分维度和最终选择。

当前评分公式：

```text
score =
  historical_success × 0.30
  + latency_score × 0.20
  + cost_score × 0.10
  + hybrid_scenario_score × 0.40

hybrid_scenario_score =
  agent_scenario_bid × 0.4
  + llm_scenario_assessment × 0.6
```

指标来源：

- `historical_success`：优先使用 Judge 业务反馈，暂无业务反馈时使用调用结果；
  结合 `success_prior_calls` 平滑。
- `latency_score`：实际耗时的 EMA；无观测值时使用 Agent Card 初始延迟。
- `cost_score`：Agent Card 手动配置的综合运行/Token 成本等级。
- `hybrid_scenario_score`：融合 Agent 确定性报价与 LLM 场景评估；LLM
  不可用时直接使用 Agent 报价。

当前限制：

- 商品召回已形成真实竞争，其他 capability 仍多数只有一个 Agent。
- Broker 尚未实现熔断和 Agent 失败后的次优候选切换。
- 当前采用请求级粗粒度归因：最终业务结果统一反馈给本次参与质量链路的 Agent，
  尚未实现 Shapley Value、对照实验等精细贡献归因。
- 成本目前是 Agent Card 的综合成本等级，不是实际 Token 账单。

### 3.4 DAG Executor

- [x] 根据依赖关系查找 ready tasks。
- [x] 同一 Wave 的任务通过 `asyncio.gather` 并行执行。
- [x] 记录每个 Wave 的任务列表。
- [x] 记录任务状态、执行 Agent、Broker 分数、耗时、输出字段和异常。
- [x] 支持执行 Planner 后续追加的新任务。

当前限制：

- 当前采用 Wave Barrier：同一 Wave 全部完成后才计算下一批任务。
- 某个快速任务完成后，其下游不能立即启动，必须等待同 Wave 的慢任务。
- 尚未实现任务超时、节点级重试、取消和优先级队列。

### 3.5 Blackboard

- [x] 为每个请求创建独立 Blackboard。
- [x] 保存任务输出。
- [x] 保存已完成任务集合。
- [x] 保存计划、Wave、Replan 和任务执行轨迹。
- [x] Agent 之间通过 Blackboard 共享结果，减少直接耦合。

当前限制：

- Blackboard 只存在于当前进程内存。
- 请求中断或进程重启后无法恢复。
- `values` 使用 `dict[str, Any]`，缺少强类型输出契约。
- 不同任务可能覆盖相同字段。
- 尚未实现版本、命名空间、访问权限和状态持久化。

### 3.6 推荐任务节点

- [x] `product_recall`：调用已有推荐召回能力。
- [x] `knowledge_retrieval`：执行公共知识库 RAG 检索。
- [x] `inventory_filter`：过滤不可售或无库存商品。
- [x] `candidate_evaluation`：结合上下文和 RAG 证据评估商品。
- [x] `recommendation_judge`：审核推荐是否满足约束。
- [x] `candidate_revision`：根据审核问题重新评估候选商品。
- [x] `recommendation_rejudge`：再次审核修正结果。
- [x] `response_synthesis`：生成最终商品排序和导购回答。

### 3.7 Replanner

- [x] 根据 Judge 的 `passed` 和 `retry_recommended` 判断是否触发 Replan。
- [x] 触发后向现有执行过程追加修正和复审任务。
- [x] 复用 Blackboard 中已经完成的结果。
- [x] 当前限制最多 Replan 一次，避免无限循环。
- [x] Replan 原因和追加任务写入执行轨迹。
- [x] Judge issues、请求约束、Blackboard 摘要和已完成任务传给 LLM。
- [x] LLM 通过 `ReplanDecision` 输出结构化 Task DAG。
- [x] 支持动态选择补充 RAG、重新召回、库存过滤、重新评估、修订、复审和回答合成能力。
- [x] 动态任务输入可向 RAG 和候选修订节点传递查询或修订指令。
- [x] 非法 capability、缺失依赖、重复 ID、循环依赖、未连接任务、
  多个终止节点和任务数超限会被安全门拒绝。
- [x] 模型不可用、输出不合规或安全检查失败时使用规则 Replanner。

当前限制：

- 当前 `max_replans` 实际只支持是否允许一次修正，还未实现通用多轮循环。
- 尚未增加 Token 预算、模型调用费用和计划最大深度限制。

### 3.8 RAG

- [x] 只有推荐链路才进入 RAG。
- [x] RAG 开启且查询非空时，在商品召回与库存过滤后执行。
- [x] 基于商品 ID、名称、品牌和品类构造候选商品证据查询。
- [x] 多个候选商品的证据检索通过 `asyncio.gather` 并行执行。
- [x] 检索结果参与候选商品评估。
- [x] 检索证据参与最终回答生成。
- [x] 返回 RAG Trace 和 citations。
- [x] RAG 服务不可用时可以跳过该任务。

### 3.9 前端执行轨迹

- [x] 返回 `orchestration_mode`。
- [x] 返回 `orchestration_trace`。
- [x] 展示 Adaptive Agent Mesh 标识。
- [x] 展示 Planner 原因。
- [x] 展示执行 Wave。
- [x] 展示任务状态、Agent、依赖、耗时和 Broker 分数。
- [x] 展示是否发生 Replan。
- [x] 展示初始计划和动态计划的 Planner 来源。
- [x] 展示 Broker 全部候选 Agent 的排名和评分维度。
- [x] 展示 LLM Replanner 回退规则计划的具体原因。

当前限制：

- Trace 在整个请求完成后一次性返回。
- 尚未通过 SSE 或 WebSocket 实时显示执行过程。
- 尚未显示实时 Agent 运行状态，当前只展示请求结束后的完整排名。

### 3.10 测试和构建

- [x] 初始 DAG 测试。
- [x] 商品召回、库存过滤与 RAG 前置依赖测试。
- [x] 候选商品证据并行检索测试。
- [x] Judge 触发 Replan 测试。
- [x] Broker 选择最高分 Agent 测试。
- [x] DAG 依赖校验测试。
- [x] 并行 Wave 执行测试。
- [x] Mesh 端到端模拟测试。
- [x] LLM Replanner 动态 DAG 测试。
- [x] 动态任务实际执行测试。
- [x] 非法 capability 回退测试。
- [x] 循环 DAG 回退测试。
- [x] 语义、画像和热度三种 Broker 场景选择测试。
- [x] 三种商品召回策略实际执行路径测试。
- [x] 统一 Agent YAML 加载测试。
- [x] Broker 权重总和校验测试。
- [x] 模型 Profile 引用校验测试。
- [x] Mesh executor 白名单测试。
- [x] Agent 动态场景报价参与 Broker 选择测试。
- [x] 历史成功率先验平滑和实际延迟更新测试。
- [x] 业务质量反馈回流与 Registry 更新测试。
- [x] 请求级、任务级评测持久化和幂等测试。
- [x] 持久化运行指标恢复和聚合评测测试。
- [x] 后端测试通过：23 个测试。
- [x] 前端 TypeScript 类型检查通过。
- [x] 前端 Vite 生产构建通过。

当前限制：

- 主要使用 Fake Agent 和单元测试。
- 尚未覆盖真实模型、真实数据库、真实 RAG 的集成测试。
- 尚未进行压力、超时、故障注入和恢复测试。

### 3.11 统一 Agent 配置

- [x] 新增版本化 `config/agents.yaml`。
- [x] `.env` 只保留模型凭证、基础设施和配置文件路径。
- [x] 使用 Pydantic 校验编排、Broker、模型 Profile、Agent Runtime 和 Agent Card。
- [x] Broker 权重由 YAML 驱动并校验总和为 1。
- [x] Mesh Agent Card 和 capability 注册由 YAML 驱动。
- [x] Mesh executor 通过代码白名单绑定，不允许配置任意动态导入。
- [x] 主推荐 Agent 的 timeout、重试、temperature 和 max_tokens 已迁移。
- [x] LLM Replanner 的模型参数已迁移。
- [x] 客服 Router、内容生成和质量校验模型参数已迁移。
- [x] 支持模型 Profile 与单 Agent 参数覆盖。
- [x] 应用启动时加载并校验配置，配置错误会快速失败。
- [x] Trace 返回配置版本和实际 Broker 权重。

当前限制：

- YAML 在应用启动时加载，暂不支持热更新。
- Registry 运行指标会从评测事件恢复；多进程实例之间仍不是实时同步。
- 配置版本已包含在持久化 Trace 中，尚未单独建立可查询的配置版本维表。
- RAG 底层模型仍使用其基础设施环境配置，不属于 Agent Card 配置。

### 3.12 业务质量反馈与持久化评测

- [x] 新增请求级 `mesh_evaluations` 持久化表。
- [x] 新增任务级 `agent_evaluations` 追加事件表。
- [x] 保存 Broker 总分、四项分数、Agent Bid、执行结果和节点延迟。
- [x] 保存初次 Judge、最终 Judge、业务成功、Replan 和 Fallback 结果。
- [x] 以“最终 Judge 通过且存在推荐商品”定义当前业务成功。
- [x] 将业务成功回流到召回、知识、库存、候选评估和修订 Agent。
- [x] Broker 有业务反馈时使用业务成功率，无反馈时回退调用成功率。
- [x] 应用启动后从持久化事件恢复调用、业务成功和延迟统计。
- [x] 提供管理员聚合接口 `GET /api/v1/agent-evaluations?days=30`。
- [x] 聚合业务成功率、Judge 通过率、Replan 率、Replan 修复率、Fallback 率。
- [x] 按 Agent 和 capability 聚合执行成功率、业务成功率、延迟、Broker 分数、
  场景报价和报价校准误差。
- [x] 持久化失败只记录告警，不阻断主推荐链路。

当前限制：

- 当前使用请求级结果进行粗粒度 Agent 归因。
- 评测查询在应用层聚合，数据量大后需要迁移到 SQL 聚合或 OLAP。
- 多 Worker 的 Broker 内存状态不会实时同步，只会在进程启动时恢复。
- 尚未接入 CTR、CVR、购买、退款等真实用户行为反馈。
- 尚未实现按模型版本、Prompt 版本和配置版本的对照分析。

## 4. 尚未完全实现的能力

### P0：LLM Planner 与 Replanner（已完成）

目标：让计划真正根据请求和失败原因动态变化。

- [x] 定义严格的 `ReplanDecision` 和 `MeshTask` Pydantic Schema。
- [x] 让 Planner 使用模型结构化输出生成任务 DAG。
- [x] 让 Replanner 读取 Judge issues、已有结果、请求约束和剩余 Replan 次数。
- [x] 根据不同问题选择重新召回、重新评估、补充 RAG 或直接终止。
- [x] 校验 capability 是否已注册并位于安全白名单。
- [x] 校验循环依赖、缺失依赖、重复 ID 和未连接任务。
- [x] 限制最大任务数和最大 Replan 次数。
- [ ] 增加最大 DAG 深度和 Token 预算限制。
- [x] Schema 或安全校验失败时回退规则 Replanner。

验收标准：

- 不同 Judge 问题能够产生不同的后续 DAG。
- 非法计划不会进入 Executor。
- LLM 输出异常时不影响主链路可用性。

### P0：Hybrid Broker 竞争（已完成）/ 故障切换（待完成）

目标：让 Broker 不只是架构占位，而是实际决定执行者。

- [x] 为 `product_recall` 注册语义、画像和热度三个候选 Agent。
- [x] 为 `candidate_evaluation` 注册快速版和高质量版 Agent。
- [x] Agent 主动返回场景匹配报价。
- [x] LLM 对候选 Agent 进行结构化场景匹配评估。
- [ ] 加入熔断状态评分。
- [x] 记录所有候选 Agent 的评分，而不只是最终选择。
- [x] 根据业务成功率和实际延迟更新并持久化运行指标。
- [ ] 支持 Agent 失败后选择次优 Agent 重试。

验收标准：

- [x] 同一 capability 在不同场景下能选择不同 Agent。
- Agent 失败后可以自动切换候选者。
- [x] 前端 Trace 能展示候选排名和评分维度。

### P0：节点级容错

目标：避免一个节点失败就从头运行 Supervisor。

- [ ] 为每个 MeshTask 增加超时配置。
- [ ] 增加最大重试次数和退避策略。
- [ ] 区分可重试错误、不可重试错误和降级错误。
- [ ] Broker 在重试时排除失败 Agent。
- [ ] RAG 失败时只降级 RAG，不回退整个 Workflow。
- [ ] 使用已有 Blackboard 数据继续执行。
- [ ] 仅在关键节点无法恢复时整体回退 Supervisor。

验收标准：

- 非关键节点失败不会导致全链路重新执行。
- 重试不会重复产生不可逆副作用。
- Trace 能显示重试、换 Agent 和降级原因。

### P1：事件驱动 DAG Executor

目标：消除 Wave Barrier，提高真实并行度。

- [ ] 每个任务完成后立即检查其下游依赖。
- [ ] 下游依赖满足后立即进入运行队列。
- [ ] 支持运行中任务取消。
- [ ] 支持最大并发数。
- [ ] 支持任务优先级和全局截止时间。
- [ ] 保留 Wave 作为展示信息，而不是执行屏障。

验收标准：

- `product_recall` 完成后可立即执行 `inventory_filter`，随后执行商品证据检索。
- 并发量受到配置限制。
- 任一任务异常不会遗留未回收协程。

### P1：强类型 Task Contract

目标：避免 Blackboard 字段冲突和运行时类型错误。

- [ ] 为每个 capability 定义输入 Schema。
- [ ] 为每个 capability 定义输出 Schema。
- [ ] Registry 注册时绑定 Schema。
- [ ] Executor 在发布 Blackboard 前校验输出。
- [ ] 为 Blackboard 增加任务命名空间。
- [ ] 禁止未授权覆盖其他任务结果。
- [ ] 增加 Schema 版本号。

验收标准：

- Agent 输出缺字段或字段类型错误时立即被识别。
- 不同任务不能意外覆盖彼此的数据。
- Trace 能展示输入输出 Schema 版本。

### P1：持久化和可恢复执行

目标：支持进程重启、任务暂停和失败恢复。

- [ ] 为 request、plan、task 和 output 建立持久化模型。
- [ ] 使用 Redis 或 PostgreSQL 保存 Blackboard checkpoint。
- [ ] 每个任务使用幂等 Task ID。
- [ ] 支持从最近成功节点恢复。
- [ ] 增加任务租约，避免多个 Worker 重复执行。
- [ ] 处理执行中进程崩溃和超时回收。

验收标准：

- 执行中重启服务后可以继续任务。
- 相同 Task ID 不会重复写入结果。
- 可查询历史计划和完整执行轨迹。

### P1：可观测性和成本治理

目标：能够解释、评估并控制每次 Mesh 执行。

- [ ] 记录每个节点的模型名称和 Prompt 版本。
- [ ] 记录输入 Token、输出 Token 和费用。
- [ ] 接入 OpenTelemetry Trace。
- [ ] 统一 request ID、task ID 和 agent ID。
- [ ] 增加成功率、P95 延迟、Replan 率和 fallback 率指标。
- [ ] 增加每次请求的 Token 和费用预算。
- [ ] 达到预算时由 Planner 选择降级或停止。

验收标准：

- 可以定位一次请求中最慢、最贵和失败的节点。
- 可以按 Agent 统计成功率和延迟。
- 超出预算时能稳定执行降级策略。

### P2：实时执行界面

目标：让用户实时看到 Agent Mesh 的运行过程。

- [ ] 后端增加 SSE 或 WebSocket 事件流。
- [ ] 推送 plan_created、task_started、task_completed、task_failed、replanned 等事件。
- [ ] 前端增量更新 DAG 和任务状态。
- [ ] 显示 Broker 候选评分。
- [ ] 显示 Judge 问题和 Replanner 追加任务。
- [ ] 支持折叠详细 Prompt、输出摘要和错误。

验收标准：

- 任务开始和结束状态无需等待最终回答即可显示。
- 断线重连后能恢复当前执行状态。

### P2：评测和稳定性

目标：证明新架构相对 Supervisor Workflow 的收益。

- [ ] 建立推荐准确性评测集。
- [ ] 对比 Workflow 与 Mesh 的质量、延迟和费用。
- [ ] 增加真实 RAG 集成测试。
- [ ] 增加数据库和库存服务集成测试。
- [ ] 增加并发压力测试。
- [ ] 增加超时、模型异常、RAG 异常和 Agent 崩溃测试。
- [ ] 评估错误 Replan 和无效循环比例。

验收标准：

- 有可重复运行的 Workflow vs Mesh 对比报告。
- 核心故障场景均有自动化测试。

## 5. 推荐开发顺序

### 阶段一：完善面试展示闭环

1. ~~LLM Replanner + JSON Schema。~~ 已完成。
2. ~~为一个 capability 注册多个真实 Agent。~~ 已完成。
3. ~~Broker 展示全部候选评分和选择理由。~~ 已完成。
4. ~~前端展示动态 Replan 的新增节点。~~ 已完成。
5. Agent 失败后由 Broker 切换 Agent。
6. 为 Candidate Evaluator 注册快速版和高质量版 Agent。

完成后可展示：

```text
Plan
  → Capability Discovery
  → Contract-Net Broker
  → Execute
  → Judge
  → Dynamic Replan
  → Agent Failover
  → Synthesize
```

### 阶段二：提高架构完整性

1. 事件驱动 Executor。
2. 强类型 Task Contract。
3. 节点超时、重试和降级。
4. Token/成本预算。
5. OpenTelemetry 可观测性。

### 阶段三：生产化

1. Blackboard 持久化。
2. Checkpoint 和恢复。
3. 多 Worker 任务租约。
4. 实时事件流。
5. 压测、故障注入和 A/B 评测。

## 6. 下次开发建议起点

建议下次直接从以下任务开始：

```text
实现 Broker 节点级故障切换：

1. 给 MeshTask 增加 timeout、max_retries 和 retry_policy。
2. Agent 执行失败后把当前 Agent 加入本任务排除集合。
3. Broker 从剩余候选中选择下一名 Agent。
4. 增加熔断状态和半开恢复机制。
5. Trace 记录每次失败、重试、换 Agent 和最终降级原因。
6. 为商品召回增加失败切换和全部失败测试。
```

## 7. 关键代码位置

- Mesh 核心：`backend/orchestrator/adaptive_mesh.py`
- 编排器选择：`backend/core/application_state.py`
- 配置：`backend/core/config.py`
- API 接入：`backend/api/sales_chat.py`
- 返回 Schema：`backend/models/schemas.py`
- 前端类型：`frontend/src/api/types.ts`
- 前端轨迹展示：`frontend/src/pages/ChatPage.vue`
- 前端样式：`frontend/src/styles/app.css`
- 持久化评测：`backend/services/agent_evaluation.py`
- 评测数据模型：`backend/database/models.py`
- 评测查询接口：`backend/api/admin.py`
- Mesh 测试：`tests/test_adaptive_mesh.py`
- 持久化评测测试：`tests/test_agent_evaluation.py`
- 环境变量示例：`.env.example`

## 8. 当前验证记录

截至 2026-09-06：

- 后端单元测试：23/23 通过。
- 前端 TypeScript 检查：通过。
- 前端生产构建：通过。
- Mesh 默认装配检查：通过。
- LLM 动态计划执行与规则回退：通过。
- Broker 三场景竞争选择与候选排名：通过。
- Broker 四指标评分、Agent Bid 和运行指标平滑：通过。
- Judge 业务反馈回流、评测持久化、重启恢复和聚合查询：通过。
- Agent YAML、模型 Profile、Broker 权重和 executor 白名单校验：通过。
- 当前变更尚未提交 Git。
