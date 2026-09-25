<script setup lang="ts">
import { ref } from "vue";

import {
  generateFaq,
  generateSalesScript,
  reviewServiceReply,
} from "../api/customerService";
import type { ServiceTaskResponse } from "../api/types";

type ToolMode = "faq" | "script" | "review";

const mode = ref<ToolMode>("faq");
const category = ref("");
const brand = ref("");
const topic = ref("");
const customerNeed = ref("");
const customerMessage = ref("");
const agentReply = ref("");
const policyContext = ref("");
const loading = ref(false);
const error = ref("");
const result = ref<ServiceTaskResponse | null>(null);

async function runTool() {
  loading.value = true;
  error.value = "";
  result.value = null;
  try {
    if (mode.value === "faq") {
      if (!topic.value.trim()) throw new Error("请输入 FAQ 主题");
      result.value = await generateFaq({
        topic: topic.value.trim(),
        category: category.value.trim(),
        brand: brand.value.trim(),
        count: 5,
        tone: "professional",
      });
    } else if (mode.value === "script") {
      if (!customerNeed.value.trim()) throw new Error("请输入客户需求");
      result.value = await generateSalesScript({
        customer_need: customerNeed.value.trim(),
        category: category.value.trim(),
        brand: brand.value.trim(),
        channel: "online",
        tone: "professional",
      });
    } else {
      if (!customerMessage.value.trim() || !agentReply.value.trim()) {
        throw new Error("请输入客户消息和待审核回复");
      }
      result.value = await reviewServiceReply({
        customer_message: customerMessage.value.trim(),
        agent_reply: agentReply.value.trim(),
        policy_context: policyContext.value.trim(),
        category: category.value.trim(),
        brand: brand.value.trim(),
      });
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "处理失败";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <section class="workspace-grid">
    <article class="panel">
      <h2>客服内容工具</h2>
      <div class="button-row service-modes">
        <button
          v-for="item in [
            { key: 'faq', label: 'FAQ 生成' },
            { key: 'script', label: '导购话术' },
            { key: 'review', label: '回复质检' },
          ]"
          :key="item.key"
          class="secondary"
          :class="{ active: mode === item.key }"
          @click="mode = item.key as ToolMode"
        >
          {{ item.label }}
        </button>
      </div>

      <div class="form-grid">
        <label>商品类目<input v-model="category" placeholder="例如：耳机" /></label>
        <label>品牌<input v-model="brand" placeholder="可选" /></label>
      </div>

      <div v-if="mode === 'faq'" class="stack compact tool-fields">
        <label>FAQ 主题<textarea v-model="topic" rows="4" placeholder="例如：降噪耳机售前常见问题" /></label>
      </div>
      <div v-else-if="mode === 'script'" class="stack compact tool-fields">
        <label>客户需求<textarea v-model="customerNeed" rows="5" placeholder="例如：预算 800 元，需要适合地铁通勤的降噪耳机" /></label>
      </div>
      <div v-else class="stack compact tool-fields">
        <label>客户消息<textarea v-model="customerMessage" rows="3" /></label>
        <label>待审核客服回复<textarea v-model="agentReply" rows="5" /></label>
        <label>补充政策背景<textarea v-model="policyContext" rows="3" placeholder="可选" /></label>
      </div>

      <p v-if="error" class="error-text" role="alert">{{ error }}</p>
      <button class="primary action-button" :disabled="loading" @click="runTool">
        {{ loading ? "处理中…" : "开始生成" }}
      </button>
    </article>

    <article class="panel">
      <div class="section-heading">
        <div>
          <h2>{{ result?.title || "客服结果" }}</h2>
        </div>
        <span v-if="result?.agent_route" class="status-pill">
          {{ result.agent_route }}
        </span>
      </div>
      <template v-if="result">
        <pre class="json-output">{{ JSON.stringify(result.content, null, 2) }}</pre>
        <p v-if="result.verifier_notes" class="notice">{{ result.verifier_notes }}</p>
        <div v-if="result.source_chunks.length" class="source-list">
          <article
            v-for="chunk in result.source_chunks"
            :key="chunk.chunk_id"
            class="source-card"
          >
            <strong>{{ chunk.filename }}</strong>
            <span>第 {{ chunk.page_number ?? "?" }} 页</span>
            <p>{{ chunk.text }}</p>
          </article>
        </div>
      </template>
      <div v-else class="empty-state">
        {{ loading ? "客服 Agent 正在检索并生成…" : "生成结果和知识库依据会显示在这里。" }}
      </div>
    </article>
  </section>
</template>
