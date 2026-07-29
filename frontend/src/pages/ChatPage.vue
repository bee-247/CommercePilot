<script setup lang="ts">
import { ref } from "vue";

import { sendSalesImage, sendSalesMessage } from "../api/chat";
import type { Product, SourceCitation } from "../api/types";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const props = defineProps<{ userId: string }>();
const sessionId = crypto.randomUUID();
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

async function send() {
  const message = draft.value.trim();
  if (!message || loading.value) return;
  messages.value.push({ role: "user", content: message });
  draft.value = "";
  loading.value = true;
  error.value = "";
  try {
    const result = selectedImage.value
      ? await sendSalesImage(
          selectedImage.value,
          message,
          sessionId,
          props.userId,
        )
      : await sendSalesMessage(message, sessionId, props.userId);
    messages.value.push({ role: "assistant", content: result.answer });
    products.value = result.recommendation.products;
    citations.value = result.citations ?? [];
    strategy.value = result.recall_strategy;
    selectedImage.value = null;
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "发送失败";
  } finally {
    loading.value = false;
  }
}

function pickImage(event: Event) {
  selectedImage.value =
    (event.target as HTMLInputElement).files?.item(0) ?? null;
}
</script>

<template>
  <section class="workspace-grid">
    <article class="panel conversation-panel">
      <div class="section-heading">
        <div>
          <span class="eyebrow">SALES AGENT</span>
          <h2>导购对话</h2>
        </div>
        <span v-if="strategy" class="status-pill">{{ strategy }}</span>
      </div>

      <div class="message-list">
        <div
          v-for="(message, index) in messages"
          :key="index"
          class="message"
          :class="message.role"
        >
          {{ message.content }}
        </div>
        <div v-if="loading" class="message assistant muted">正在理解需求并召回商品…</div>
      </div>

      <p v-if="error" class="error-text">{{ error }}</p>
      <div class="composer">
        <textarea
          v-model="draft"
          rows="2"
          placeholder="例如：预算 800 元，想买适合地铁通勤的降噪耳机"
          @keydown.enter.exact.prevent="send"
        />
        <div class="composer-actions">
          <label class="upload-chip">
            {{ selectedImage ? selectedImage.name : "添加图片" }}
            <input type="file" accept="image/*" @change="pickImage" />
          </label>
          <button class="primary" :disabled="loading" @click="send">发送</button>
        </div>
      </div>
    </article>

    <aside class="panel result-panel">
      <div class="section-heading">
        <div>
          <span class="eyebrow">RECOMMENDATIONS</span>
          <h2>商品结果</h2>
        </div>
        <span class="count-badge">{{ products.length }}</span>
      </div>
      <div v-if="products.length" class="product-list">
        <article v-for="product in products" :key="product.product_id" class="product-card">
          <div class="product-meta">
            <span>{{ product.category }}</span>
            <span>库存 {{ product.stock }}</span>
          </div>
          <h3>{{ product.name }}</h3>
          <p>{{ product.description || "暂无商品描述" }}</p>
          <strong>¥{{ product.price.toFixed(2) }}</strong>
        </article>
      </div>
      <div v-else class="empty-state">推荐商品会显示在这里。</div>
      <div v-if="citations.length" class="citation-list">
        <span class="eyebrow">KNOWLEDGE EVIDENCE</span>
        <article
          v-for="(citation, index) in citations"
          :key="`${citation.filename}-${index}`"
          class="source-card"
        >
          <strong>{{ citation.filename }}</strong>
          <span>第 {{ citation.page_number ?? "?" }} 页</span>
          <p>{{ citation.text }}</p>
        </article>
      </div>
    </aside>
  </section>
</template>
