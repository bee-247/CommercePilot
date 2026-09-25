<script setup lang="ts">
import { onMounted, ref } from "vue";

import {
  getExperiments,
  getHealth,
  getMetrics,
  getReadiness,
  indexProducts,
} from "../api/admin";
import type { ReadinessResponse } from "../api/admin";

const health = ref<{ status: string; model: string } | null>(null);
const readiness = ref<ReadinessResponse | null>(null);
const metrics = ref<Record<string, unknown>>({});
const experiments = ref<Record<string, unknown>>({});
const indexResult = ref<Record<string, unknown> | null>(null);
const indexing = ref(false);
const error = ref("");

async function refresh() {
  error.value = "";
  try {
    [health.value, readiness.value, metrics.value, experiments.value] =
      await Promise.all([
      getHealth(),
      getReadiness(),
      getMetrics(),
      getExperiments(),
    ]);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "状态读取失败";
  }
}

async function rebuildProductIndex() {
  indexing.value = true;
  error.value = "";
  try {
    indexResult.value = await indexProducts();
    await refresh();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "商品索引失败";
  } finally {
    indexing.value = false;
  }
}

onMounted(refresh);
</script>

<template>
  <section class="stack">
    <article class="panel status-overview">
      <div>
        <h2>运行概览</h2>
      </div>
      <div class="health-card" :class="{ online: health?.status === 'healthy' }">
        <span class="health-dot"></span>
        <div>
          <strong>{{ health?.status || "unknown" }}</strong>
          <small>{{ health?.model || "模型未报告" }}</small>
        </div>
      </div>
      <button class="secondary" @click="refresh">刷新状态</button>
      <button class="primary" :disabled="indexing" @click="rebuildProductIndex">
        {{ indexing ? "索引中…" : "重建商品索引" }}
      </button>
    </article>
    <article class="panel">
      <h2>依赖就绪状态</h2>
      <div v-if="readiness" class="readiness-grid">
        <article
          v-for="(component, name) in readiness.components"
          :key="name"
          :class="{ online: component.status === 'ready' }"
        >
          <span class="health-dot"></span>
          <div>
            <strong>{{ name }}</strong>
            <small>{{ component.reason || component.status }}</small>
          </div>
        </article>
      </div>
    </article>
    <div class="workspace-grid">
      <article class="panel">
        <h2>Agent 与业务指标</h2>
        <p v-if="error" class="error-text" role="alert">{{ error }}</p>
        <pre class="json-output">{{ JSON.stringify(metrics, null, 2) }}</pre>
      </article>
      <article class="panel">
        <h2>A/B 实验</h2>
        <pre class="json-output">{{ JSON.stringify(experiments, null, 2) }}</pre>
        <p v-if="indexResult" class="notice">
          最近索引结果：{{ JSON.stringify(indexResult) }}
        </p>
      </article>
    </div>
  </section>
</template>
