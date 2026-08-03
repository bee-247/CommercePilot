<script setup lang="ts">
import { onMounted, ref } from "vue";

import {
  deleteDocument,
  getDeleteJob,
  getUploadJob,
  listDocuments,
  uploadDocument,
} from "../api/documents";
import type { DocumentInfo, DocumentJob } from "../api/types";

const documents = ref<DocumentInfo[]>([]);
const selectedFile = ref<File | null>(null);
const category = ref("");
const brand = ref("");
const loading = ref(false);
const message = ref("");
const error = ref("");
const activeJobs = ref<DocumentJob[]>([]);

async function refresh() {
  loading.value = true;
  error.value = "";
  try {
    documents.value = await listDocuments();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "加载文档失败";
  } finally {
    loading.value = false;
  }
}

function pickFile(event: Event) {
  selectedFile.value =
    (event.target as HTMLInputElement).files?.item(0) ?? null;
}

async function upload() {
  if (!selectedFile.value) {
    error.value = "请先选择 PDF、Word 或 Excel 文件";
    return;
  }
  loading.value = true;
  error.value = "";
  try {
    const result = await uploadDocument(selectedFile.value, {
      category: category.value,
      brand: brand.value,
      business_line: "customer_service",
      document_type: "product",
    });
    message.value = `${result.message}（任务 ${result.job_id.slice(0, 8)}）`;
    selectedFile.value = null;
    await pollJob(result.job_id, "upload");
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "上传失败";
  } finally {
    loading.value = false;
  }
}

async function remove(filename: string) {
  if (!window.confirm(`确认删除 ${filename} 的知识库索引吗？`)) return;
  loading.value = true;
  error.value = "";
  try {
    const result = await deleteDocument(filename);
    message.value = result.message;
    await pollJob(result.job_id, "delete");
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "删除失败";
  } finally {
    loading.value = false;
  }
}

async function pollJob(jobId: string, type: "upload" | "delete") {
  for (let attempt = 0; attempt < 300; attempt += 1) {
    const job =
      type === "upload"
        ? await getUploadJob(jobId)
        : await getDeleteJob(jobId);
    const index = activeJobs.value.findIndex((item) => item.job_id === jobId);
    if (index >= 0) {
      activeJobs.value[index] = job;
    } else {
      activeJobs.value.push(job);
    }
    if (job.status === "completed" || job.status === "failed") {
      message.value = job.message;
      await refresh();
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
  error.value = "任务仍在处理，请稍后刷新资料库查看结果";
}

onMounted(refresh);
</script>

<template>
  <section class="stack">
    <article class="panel upload-panel">
      <div class="section-heading">
        <div>
          <span class="eyebrow">INGESTION</span>
          <h2>上传知识资料</h2>
        </div>
      </div>
      <div class="form-grid">
        <label>商品类目<input v-model="category" placeholder="例如：耳机" /></label>
        <label>品牌<input v-model="brand" placeholder="例如：Acme" /></label>
        <label class="file-field">
          文件
          <input type="file" accept=".pdf,.doc,.docx,.xls,.xlsx" @change="pickFile" />
        </label>
        <button class="primary" :disabled="loading" @click="upload">开始后台入库</button>
      </div>
      <p v-if="message" class="success-text">{{ message }}</p>
      <p v-if="error" class="error-text">{{ error }}</p>
      <div v-if="activeJobs.length" class="job-list">
        <article v-for="job in activeJobs" :key="job.job_id">
          <div>
            <strong>{{ job.filename }}</strong>
            <span>{{ job.message }}</span>
          </div>
          <div class="progress-track">
            <span
              :style="{
                width: `${Math.max(...job.steps.map((step) => step.percent), 0)}%`,
              }"
            ></span>
          </div>
        </article>
      </div>
    </article>

    <article class="panel">
      <div class="section-heading">
        <div>
          <span class="eyebrow">KNOWLEDGE BASE</span>
          <h2>资料库</h2>
        </div>
        <button class="secondary" :disabled="loading" @click="refresh">刷新</button>
      </div>
      <div v-if="documents.length" class="document-table">
        <div class="table-row table-head">
          <span>文件</span><span>范围</span><span>分块</span><span>状态</span><span></span>
        </div>
        <div v-for="document in documents" :key="document.filename" class="table-row">
          <div>
            <strong>{{ document.display_name || document.filename }}</strong>
            <small>{{ document.category || "未分类" }} · {{ document.brand || "未指定品牌" }}</small>
          </div>
          <span>{{ document.visibility || "public" }}</span>
          <span>{{ document.chunk_count }}</span>
          <span class="status-pill">{{ document.status || "unknown" }}</span>
          <button
            v-if="document.is_owner"
            class="danger-link"
            @click="remove(document.filename)"
          >
            删除
          </button>
        </div>
      </div>
      <div v-else class="empty-state">{{ loading ? "正在加载…" : "还没有可见资料。" }}</div>
    </article>
  </section>
</template>
