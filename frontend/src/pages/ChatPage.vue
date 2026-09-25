<script setup lang="ts">
import { nextTick, onMounted, ref } from "vue";

import {
  deleteSalesChatSession,
  getSalesChatSessionMessages,
  listSalesChatSessions,
  sendSalesImage,
  streamSalesMessage,
  type SalesProgressEvent,
} from "../api/chat";
import type {
  AgentMeshTrace,
  Product,
  SalesChatResponse,
  SalesChatSession,
  SourceCitation,
} from "../api/types";

interface Message {
  role: "user" | "assistant";
  content: string;
  kind?: "text" | "progress";
  steps?: SalesProgressEvent[];
  progressStatus?: "running" | "completed" | "failed";
  elapsedMs?: number;
  totalTokens?: number;
  collapsed?: boolean;
}

const props = defineProps<{ userId: string }>();
const sessionId = ref<string>(crypto.randomUUID());
const draft = ref("");
const loading = ref(false);
const error = ref("");
const messages = ref<Message[]>([
  {
    role: "assistant",
    content: "告诉我你的预算、使用场景或偏好，我会结合商品库给出推荐。",
  },
]);
const products = ref<Product[]>([]);
const citations = ref<SourceCitation[]>([]);
const selectedImage = ref<File | null>(null);
const strategy = ref("");
const orchestrationMode = ref("");
const meshTrace = ref<AgentMeshTrace | null>(null);
const tokenUsage = ref<Record<string, number>>({});
const requestLatencyMs = ref<number | null>(null);
const clientLatencyMs = ref<number | null>(null);
const activeProgress = ref<Message | null>(null);
const messageList = ref<HTMLElement | null>(null);
const sessions = ref<SalesChatSession[]>([]);
const historyLoading = ref(false);
const deletingSessionId = ref("");

const welcomeMessage: Message = {
  role: "assistant",
  content: "告诉我你的预算、使用场景或偏好，我会结合商品库给出推荐。",
};

function resetResultPanels() {
  products.value = [];
  citations.value = [];
  strategy.value = "";
  orchestrationMode.value = "";
  meshTrace.value = null;
  tokenUsage.value = {};
  requestLatencyMs.value = null;
  clientLatencyMs.value = null;
  error.value = "";
}

function newConversation() {
  sessionId.value = crypto.randomUUID();
  messages.value = [{ ...welcomeMessage }];
  resetResultPanels();
}

async function refreshSalesSessions() {
  try {
    sessions.value = await listSalesChatSessions(props.userId);
  } catch {
    sessions.value = [];
  }
}

async function openConversation(session: SalesChatSession) {
  if (loading.value) return;
  historyLoading.value = true;
  try {
    const history = await getSalesChatSessionMessages(
      props.userId,
      session.session_id,
    );
    sessionId.value = session.session_id;
    messages.value = history.length ? history : [{ ...welcomeMessage }];
    resetResultPanels();
    void scrollConversation();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "历史对话加载失败";
  } finally {
    historyLoading.value = false;
  }
}

async function deleteConversation(session: SalesChatSession) {
  if (loading.value || deletingSessionId.value) return;
  if (!window.confirm(`确定删除对话“${session.preview || "新对话"}”吗？`)) {
    return;
  }
  deletingSessionId.value = session.session_id;
  error.value = "";
  try {
    const deleted = await deleteSalesChatSession(
      props.userId,
      session.session_id,
    );
    if (!deleted) {
      throw new Error("该历史对话不存在或已经被删除");
    }
    sessions.value = sessions.value.filter(
      (item) => item.session_id !== session.session_id,
    );
    if (session.session_id === sessionId.value) {
      newConversation();
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "历史对话删除失败";
  } finally {
    deletingSessionId.value = "";
  }
}

function formatSessionTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? ""
    : new Intl.DateTimeFormat("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}

async function scrollConversation() {
  await nextTick();
  if (messageList.value) {
    messageList.value.scrollTop = messageList.value.scrollHeight;
  }
}

function updateProgress(event: SalesProgressEvent) {
  const progress = activeProgress.value;
  if (!progress?.steps) return;
  const index = progress.steps.findIndex((step) => step.stage === event.stage);
  if (index >= 0) {
    progress.steps.splice(index, 1, event);
  } else {
    progress.steps.push(event);
  }
  progress.elapsedMs = event.elapsed_ms;
  void scrollConversation();
}

function formatDuration(value?: number | null) {
  if (value === undefined || value === null) return "-";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(1)} ms`;
}

function formatTokens(value?: number) {
  return new Intl.NumberFormat("zh-CN").format(value ?? 0);
}

function shortProductName(value: string) {
  const name = value.trim();
  return name.length > 24 ? `${name.slice(0, 24)}…` : name;
}

async function send() {
  const message = draft.value.trim();
  if (!message || loading.value) return;
  messages.value.push({ role: "user", content: message });
  draft.value = "";
  loading.value = true;
  error.value = "";
  meshTrace.value = null;
  tokenUsage.value = {};
  requestLatencyMs.value = null;
  clientLatencyMs.value = null;
  const requestStarted = performance.now();
  const progressMessage: Message = {
    role: "assistant",
    content: "",
    kind: "progress",
    steps: [],
    progressStatus: "running",
    elapsedMs: 0,
  };
  messages.value.push(progressMessage);
  activeProgress.value = progressMessage;
  void scrollConversation();
  const elapsedTimer = window.setInterval(() => {
    if (activeProgress.value?.progressStatus === "running") {
      activeProgress.value.elapsedMs = performance.now() - requestStarted;
    }
  }, 100);
  try {
    let result: SalesChatResponse;
    if (selectedImage.value) {
      updateProgress({
        type: "progress",
        stage: "image_chat",
        label: "分析图片并执行推荐流程",
        status: "running",
        elapsed_ms: 0,
      });
      result = await sendSalesImage(
        selectedImage.value,
        message,
        sessionId.value,
        props.userId,
      );
      updateProgress({
        type: "progress",
        stage: "image_chat",
        label: "分析图片并执行推荐流程",
        status: "completed",
        elapsed_ms: performance.now() - requestStarted,
        latency_ms: result.request_latency_ms,
      });
    } else {
      result = await streamSalesMessage(
        message,
        sessionId.value,
        props.userId,
        updateProgress,
      );
    }
    products.value = result.recommendation.products;
    citations.value = result.citations ?? [];
    strategy.value = result.recall_strategy;
    orchestrationMode.value = result.recommendation.orchestration_mode ?? "workflow";
    meshTrace.value = result.recommendation.orchestration_trace ?? null;
    tokenUsage.value = result.token_usage ?? {};
    requestLatencyMs.value = result.request_latency_ms ?? null;
    clientLatencyMs.value = performance.now() - requestStarted;
    if (activeProgress.value) {
      activeProgress.value.progressStatus = "completed";
      activeProgress.value.elapsedMs = clientLatencyMs.value;
      activeProgress.value.totalTokens = result.token_usage?.total_tokens ?? 0;
      activeProgress.value.collapsed = true;
    }
    messages.value.push({ role: "assistant", content: result.answer });
    void scrollConversation();
    void refreshSalesSessions();
    selectedImage.value = null;
  } catch (cause) {
    clientLatencyMs.value = performance.now() - requestStarted;
    if (activeProgress.value) {
      activeProgress.value.progressStatus = "failed";
      activeProgress.value.elapsedMs = clientLatencyMs.value;
    }
    error.value = cause instanceof Error ? cause.message : "发送失败";
  } finally {
    window.clearInterval(elapsedTimer);
    activeProgress.value = null;
    loading.value = false;
  }
}

function pickImage(event: Event) {
  selectedImage.value =
    (event.target as HTMLInputElement).files?.item(0) ?? null;
}

function usePrompt(prompt: string) {
  draft.value = prompt;
  void nextTick(() => document.getElementById("sales-message")?.focus());
}

onMounted(refreshSalesSessions);
</script>

<template>
  <div class="stack">
    <section class="workspace-grid chat-workspace">
      <aside class="panel history-panel">
        <div class="section-heading">
          <div>
            <h2>历史对话</h2>
          </div>
          <button class="history-new-button" :disabled="loading" @click="newConversation">
            新对话
          </button>
        </div>
        <div class="history-list" :aria-busy="historyLoading">
          <span v-if="historyLoading" class="muted">正在读取…</span>
          <div
            v-for="item in sessions"
            :key="item.session_id"
            class="history-row"
            :class="{ active: item.session_id === sessionId }"
          >
            <button
              class="history-item"
              :class="{ active: item.session_id === sessionId }"
              :disabled="loading || historyLoading || Boolean(deletingSessionId)"
              @click="openConversation(item)"
            >
              <strong>{{ item.preview || "新对话" }}</strong>
              <span>
                {{ item.message_count }} 条 / {{ formatSessionTime(item.updated_at) }}
              </span>
            </button>
            <button
              type="button"
              class="history-delete-button"
              :disabled="loading || historyLoading || Boolean(deletingSessionId)"
              :aria-label="`删除对话：${item.preview || '新对话'}`"
              title="删除对话"
              @click.stop="deleteConversation(item)"
            >
              {{ deletingSessionId === item.session_id ? "…" : "删除" }}
            </button>
          </div>
          <div v-if="!historyLoading && !sessions.length" class="history-empty">
            <span class="empty-state-icon" aria-hidden="true">↗</span>
            <span>暂无历史对话</span>
            <button type="button" class="text-button" @click="newConversation">
              开始一段新对话
            </button>
          </div>
        </div>
      </aside>

      <article class="panel conversation-panel">
      <div class="section-heading">
        <div>
          <h2>导购对话</h2>
        </div>
        <span v-if="strategy" class="status-pill">{{ strategy }}</span>
      </div>

      <div
        ref="messageList"
        class="message-list"
        aria-live="polite"
        :aria-busy="loading"
      >
        <div
          v-for="(message, index) in messages"
          :key="index"
          class="message"
          :class="[
            message.role,
            { 'progress-message': message.kind === 'progress' },
          ]"
        >
          <template v-if="message.kind === 'progress'">
            <div class="agent-progress-header">
              <strong>
                {{
                  message.progressStatus === "running"
                    ? "Agent 正在处理"
                    : message.progressStatus === "completed"
                      ? "Agent 处理完成"
                      : "Agent 处理失败"
                }}
              </strong>
              <div class="agent-progress-actions">
                <span>
                  {{ formatDuration(message.elapsedMs) }}
                  <template v-if="message.totalTokens !== undefined">
                    / {{ formatTokens(message.totalTokens) }} Token
                  </template>
                </span>
                <button
                  v-if="message.progressStatus !== 'running'"
                  type="button"
                  @click="message.collapsed = !message.collapsed"
                >
                  {{ message.collapsed ? "展开" : "收起" }}
                </button>
              </div>
            </div>
            <div v-show="!message.collapsed" class="agent-progress-body">
              <div class="agent-progress-steps">
                <article
                  v-for="step in message.steps"
                  :key="step.stage"
                  :class="step.status"
                >
                  <i aria-hidden="true" />
                  <div>
                    <strong>{{ step.label }}</strong>
                    <small v-if="step.agent_id">{{ step.agent_id }}</small>
                    <small v-if="step.error" class="progress-error">{{ step.error }}</small>
                  </div>
                  <span>
                    {{
                      step.status === "running"
                        ? "执行中"
                        : step.status === "failed"
                          ? `失败 / ${formatDuration(step.latency_ms)}`
                          : formatDuration(step.latency_ms)
                    }}
                  </span>
                </article>
              </div>
              <div v-if="message.progressStatus !== 'running'" class="agent-progress-total">
                <span>总耗时 {{ formatDuration(message.elapsedMs) }}</span>
                <span>Token {{ formatTokens(message.totalTokens) }}</span>
              </div>
            </div>
          </template>
          <template v-else>{{ message.content }}</template>
        </div>
      </div>

      <p v-if="error" class="error-text" role="alert">{{ error }}</p>
      <div class="composer">
        <label class="sr-only" for="sales-message">描述你的购买需求</label>
        <textarea
          id="sales-message"
          v-model="draft"
          rows="2"
          placeholder="例如：预算 800 元，想买适合地铁通勤的降噪耳机"
          @keydown.enter.exact.prevent="send"
        />
        <div class="composer-actions">
          <label class="upload-chip">
            {{ selectedImage ? selectedImage.name : "添加图片" }}
            <input type="file" accept="image/*" aria-label="添加商品图片" @change="pickImage" />
          </label>
          <button class="primary" :disabled="loading" @click="send">
            {{ loading ? "处理中…" : "发送" }}
          </button>
        </div>
      </div>
      </article>

      <aside class="panel result-panel">
      <div class="section-heading">
        <div>
          <h2>商品结果</h2>
        </div>
        <span class="count-badge">{{ products.length }}</span>
      </div>
      <div v-if="products.length" class="product-list">
        <details
          v-for="product in products"
          :key="product.product_id"
          class="product-card"
        >
          <summary class="product-summary" :title="product.name">
            <h3>{{ shortProductName(product.name) }}</h3>
            <span class="product-summary-footer">
              <strong>¥{{ product.price.toFixed(2) }}</strong>
              <small>查看详情</small>
            </span>
          </summary>
          <div class="product-detail">
            <h4>{{ product.name }}</h4>
            <div class="product-meta">
              <span>{{ product.category }}</span>
              <span>库存 {{ product.stock }}</span>
            </div>
            <dl class="product-facts">
              <div>
                <dt>品牌</dt>
                <dd>{{ product.brand || "未标注" }}</dd>
              </div>
              <div>
                <dt>商品 ID</dt>
                <dd>{{ product.product_id }}</dd>
              </div>
            </dl>
            <span class="label">完整介绍</span>
            <p>{{ product.description || "暂无商品描述" }}</p>
          </div>
        </details>
      </div>
      <div v-else class="empty-state result-empty-state">
        <div class="product-empty-art" aria-hidden="true">
          <span class="product-art-card product-art-card-back" />
          <span class="product-art-card product-art-card-front" />
          <span class="product-art-search" />
        </div>
        <strong>准备好为你挑选商品</strong>
        <span>从一个具体需求开始，推荐会出现在这里。</span>
        <div class="prompt-chips">
          <button type="button" @click="usePrompt('预算 500 元以内，想买适合通勤的耳机')">
            通勤耳机
          </button>
          <button type="button" @click="usePrompt('想买一些适合周末看电影时吃的休闲零食')">
            消遣零食
          </button>
          <button type="button" @click="usePrompt('想买一些适合居家使用、提升生活品质的实用好物')">
            居家好物
          </button>
        </div>
      </div>
      <div v-if="citations.length" class="citation-list">
        <h3 class="subsection-title">知识库依据</h3>
        <article
          v-for="(citation, index) in citations"
          :key="`${citation.filename}-${index}`"
          class="source-card"
        >
          <strong>{{ citation.filename }}</strong>
          <span>第 {{ citation.page_number ?? "?" }} 页</span>
          <span v-if="citation.target_product_name || citation.target_product_id">
            对应商品：{{ citation.target_product_name || citation.target_product_id }}
          </span>
          <p>{{ citation.text }}</p>
        </article>
      </div>
      </aside>
    </section>

    <section
      v-if="
        meshTrace ||
        tokenUsage.total_tokens ||
        requestLatencyMs !== null ||
        clientLatencyMs !== null
      "
      class="panel mesh-panel"
    >
      <div class="section-heading">
        <div>
          <h2>本次执行流程</h2>
        </div>
        <span class="status-pill">{{ orchestrationMode }}</span>
      </div>

      <div class="execution-metrics">
        <article>
          <span>前端总耗时</span>
          <strong>{{ formatDuration(clientLatencyMs) }}</strong>
        </article>
        <article>
          <span>后端总耗时</span>
          <strong>{{ formatDuration(requestLatencyMs) }}</strong>
        </article>
        <article>
          <span>编排耗时</span>
          <strong>{{ formatDuration(meshTrace?.total_latency_ms) }}</strong>
        </article>
        <article class="token-total">
          <span>估算 Token</span>
          <strong>{{ formatTokens(tokenUsage.total_tokens) }}</strong>
        </article>
      </div>

      <div v-if="tokenUsage.total_tokens" class="token-breakdown">
        <span>输入 {{ formatTokens(tokenUsage.input_tokens) }}</span>
        <span>输出 {{ formatTokens(tokenUsage.output_tokens) }}</span>
        <span>Agent 轨迹 {{ formatTokens(tokenUsage.agent_trace_tokens) }}</span>
        <span>记忆上下文 {{ formatTokens(tokenUsage.memory_tokens) }}</span>
        <small>Token 为本地估算值，实际计费以模型服务商账单为准。</small>
      </div>

      <div v-if="meshTrace" class="mesh-summary">
        <span>{{ meshTrace.waves.length }} 个执行波次</span>
        <span>{{ meshTrace.tasks.length }} 个动态任务</span>
        <span>{{ meshTrace.replan.triggered ? "已触发回复修订" : meshTrace.evaluation?.final_judge_passed === true ? "最终回复审核通过" : meshTrace.evaluation?.final_judge_passed === false ? "最终回复审核未通过" : "未触发回复修订" }}</span>
        <span v-if="meshTrace.evaluation">
          业务质量{{ meshTrace.evaluation.business_success ? "通过" : "未通过" }} /
          {{ meshTrace.evaluation.persisted ? "已持久化" : "未持久化" }}
        </span>
      </div>

      <div v-if="meshTrace?.plans.length" class="mesh-plan-list">
        <p
          v-for="(plan, index) in meshTrace.plans"
          :key="`${plan.phase}-${index}`"
          class="mesh-plan-reason"
        >
          {{ plan.planner || "Planner" }}：{{ plan.reason }}
          <small v-if="plan.fallback_reason">回退原因：{{ plan.fallback_reason }}</small>
        </p>
      </div>

      <div v-if="meshTrace" class="mesh-waves">
        <article v-for="(wave, index) in meshTrace.waves" :key="index">
          <strong>Wave {{ index + 1 }}</strong>
          <span>{{ wave.join(" / ") }}</span>
        </article>
      </div>

      <div v-if="meshTrace" class="mesh-task-grid">
        <article
          v-for="task in meshTrace.tasks"
          :key="task.task_id"
          class="mesh-task"
          :class="task.status"
        >
          <div class="mesh-task-header">
            <strong>{{ task.task_id }}</strong>
            <span>{{ formatDuration(task.latency_ms) }}</span>
          </div>
          <p>{{ task.agent_id }}</p>
          <small v-if="task.strategy">
            执行策略：{{ task.strategy.mode }}
            <template v-if="task.strategy.reason"> · {{ task.strategy.reason }}</template>
          </small>
          <small>
            Broker {{ task.broker_score.toFixed(3) }} /
            依赖 {{ task.depends_on.length ? task.depends_on.join(", ") : "无" }}
          </small>
          <small v-if="task.broker_mode">
            路由模式：{{ task.broker_mode }}
          </small>
          <small v-if="task.broker_fallback_reason">
            Broker 降级：{{ task.broker_fallback_reason }}
          </small>
          <small v-if="task.business_success !== undefined && task.business_success !== null">
            业务质量反馈：{{ task.business_success ? "通过" : "未通过" }}
          </small>
          <details v-if="task.broker_candidates && task.broker_candidates.length > 1">
            <summary>Broker 候选排名</summary>
            <ol class="broker-ranking">
              <li
                v-for="candidate in task.broker_candidates"
                :key="candidate.agent_id"
                :class="{ selected: candidate.selected }"
              >
                <span>
                  {{ candidate.agent_id }}
                  <small>
                    成功 {{ candidate.components.historical_success.toFixed(2) }} /
                    延迟 {{ candidate.components.latency.toFixed(2) }} /
                    成本 {{ candidate.components.cost.toFixed(2) }} /
                    场景 {{ candidate.components.scenario.toFixed(2) }}
                  </small>
                  <small>{{ candidate.bid.reason }}</small>
                  <small v-if="candidate.llm_assessment">
                    LLM 匹配 {{ candidate.llm_assessment.scenario_score.toFixed(2) }} /
                    {{ candidate.llm_assessment.reason }}
                  </small>
                </span>
                <strong>{{ candidate.score.toFixed(3) }}</strong>
              </li>
            </ol>
          </details>
        </article>
      </div>

      <p v-if="meshTrace?.replan.triggered" class="mesh-replan">
        Replanner（{{ meshTrace.replan.planner || "rule" }}）：
        {{ meshTrace.replan.reason }}
        <small v-if="meshTrace.replan.fallback_reason">
          结构化计划回退原因：{{ meshTrace.replan.fallback_reason }}
        </small>
      </p>
      <p v-if="meshTrace?.error" class="error-text">
        Mesh 已降级：{{ meshTrace.error }}
      </p>
    </section>
  </div>
</template>
