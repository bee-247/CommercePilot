<script setup lang="ts">
import { onMounted, ref } from "vue";

import { streamRagMessage } from "../api/rag";
import {
  deleteSession,
  getSessionMessages,
  listSessions,
} from "../api/sessions";
import type { RagSession, RagTrace } from "../api/types";

const sessionId = ref<string>(crypto.randomUUID());
const question = ref("");
const answer = ref("");
const trace = ref<RagTrace | null>(null);
const route = ref("");
const loading = ref(false);
const error = ref("");
const steps = ref<Array<{ label: string; detail?: string }>>([]);
const sessions = ref<RagSession[]>([]);
const tokenUsage = ref<Record<string, number>>({});

async function ask() {
  const message = question.value.trim();
  if (!message || loading.value) return;
  loading.value = true;
  error.value = "";
  answer.value = "";
  trace.value = null;
  steps.value = [];
  try {
    await streamRagMessage(message, sessionId.value, (event) => {
      if (event.type === "content") {
        answer.value += String(event.content ?? "");
      } else if (event.type === "rag_step") {
        const step = event.step as { label?: string; detail?: string };
        steps.value.push({
          label: step.label ?? "检索步骤",
          detail: step.detail,
        });
      } else if (event.type === "trace") {
        trace.value = (event.rag_trace as RagTrace) ?? null;
      } else if (event.type === "agent_route") {
        route.value = String(event.agent_route ?? "");
      } else if (event.type === "token_usage") {
        tokenUsage.value =
          (event.token_usage as Record<string, number>) ?? {};
      } else if (event.type === "error") {
        error.value = String(event.content ?? "生成失败");
      }
    });
    await refreshSessions();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "检索失败";
  } finally {
    loading.value = false;
  }
}

async function refreshSessions() {
  try {
    sessions.value = await listSessions();
  } catch {
    sessions.value = [];
  }
}

async function openSession(value: RagSession) {
  sessionId.value = value.session_id;
  const messages = await getSessionMessages(value.session_id);
  answer.value = messages
    .map((message) => `${message.type === "human" ? "用户" : "助手"}：${message.content}`)
    .join("\n\n");
  trace.value =
    [...messages].reverse().find((message) => message.rag_trace)?.rag_trace ??
    null;
}

async function removeSession(value: RagSession) {
  await deleteSession(value.session_id);
  if (sessionId.value === value.session_id) {
    sessionId.value = crypto.randomUUID();
    answer.value = "";
    trace.value = null;
  }
  await refreshSessions();
}

onMounted(refreshSessions);
</script>

<template>
  <section class="workspace-grid">
    <article class="panel conversation-panel">
      <div class="section-heading">
        <div>
          <span class="eyebrow">KNOWLEDGE CHAT</span>
          <h2>RAG 问答</h2>
        </div>
        <span v-if="route" class="status-pill">{{ route }}</span>
      </div>
      <div v-if="sessions.length" class="session-strip">
        <article v-for="item in sessions" :key="item.session_id">
          <button class="text-button" @click="openSession(item)">
            {{ item.session_id.slice(0, 12) }} · {{ item.message_count }} 条
          </button>
          <button class="danger-link" @click="removeSession(item)">删除</button>
        </article>
      </div>
      <textarea
        v-model="question"
        class="question-input"
        rows="4"
        placeholder="输入需要从知识库中查证的问题"
        @keydown.enter.exact.prevent="ask"
      />
      <div class="button-row">
        <button class="primary" :disabled="loading" @click="ask">
          {{ loading ? "检索中…" : "检索并回答" }}
        </button>
      </div>
      <p v-if="error" class="error-text">{{ error }}</p>
      <div v-if="answer" class="answer-block">{{ answer }}</div>
      <div v-else class="empty-state">回答及引用依据会显示在这里。</div>
    </article>

    <aside class="panel trace-panel">
      <div class="section-heading">
        <div>
          <span class="eyebrow">RAG TRACE</span>
          <h2>检索轨迹</h2>
        </div>
      </div>
      <template v-if="trace">
        <dl class="trace-facts">
          <div><dt>模式</dt><dd>{{ trace.retrieval_mode || "—" }}</dd></div>
          <div><dt>阶段</dt><dd>{{ trace.retrieval_stage || "—" }}</dd></div>
          <div><dt>重写</dt><dd>{{ trace.rewrite_strategy || "未触发" }}</dd></div>
          <div><dt>Rerank</dt><dd>{{ trace.rerank_applied ? "已应用" : "未应用" }}</dd></div>
          <div><dt>Auto merge</dt><dd>{{ trace.auto_merge_applied ? "已应用" : "未应用" }}</dd></div>
        </dl>
        <div class="source-list">
          <article
            v-for="(chunk, index) in trace.retrieved_chunks || []"
            :key="`${chunk.filename}-${index}`"
            class="source-card"
          >
            <strong>{{ chunk.filename }}</strong>
            <span>第 {{ chunk.page_number ?? "?" }} 页</span>
            <p>{{ chunk.text }}</p>
          </article>
        </div>
      </template>
      <div v-if="steps.length" class="trace-steps">
        <article v-for="(step, index) in steps" :key="index">
          <strong>{{ step.label }}</strong>
          <span>{{ step.detail }}</span>
        </article>
      </div>
      <p v-if="tokenUsage.total_tokens" class="muted">
        本会话 Token：{{ tokenUsage.total_tokens }}
      </p>
      <div v-else class="empty-state">完成一次问答后可查看检索细节。</div>
    </aside>
  </section>
</template>
