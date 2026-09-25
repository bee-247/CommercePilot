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
const fileInput = ref<HTMLInputElement | null>(null);
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
      business_line: "customer_service",
      document_type: "general",
    });
    message.value = `${result.message}（任务 ${result.job_id.slice(0, 8)}）`;
    selectedFile.value = null;
    if (fileInput.value) fileInput.value.value = "";
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
          <h2>上传知识文档</h2>
          <p class="muted">用于平台客服回答的通用资料，例如平台规则、常见问题和售后流程。</p>
        </div>
      </div>
      <div class="form-grid">
        <label class="file-field">
          选择文档
          <input
            ref="fileInput"
            type="file"
            accept=".pdf,.doc,.docx,.xls,.xlsx"
            aria-describedby="document-format-hint"
            :disabled="loading"
            @change="pickFile"
          />
          <span id="document-format-hint" class="muted">支持 PDF、Word、Excel，每次上传一个文件。</span>
        </label>
        <button class="primary" :disabled="loading || !selectedFile" @click="upload">上传并入库</button>
      </div>
      <p v-if="message" class="success-text">{{ message }}</p>
      <p v-if="error" class="error-text" role="alert">{{ error }}</p>
      <div v-if="activeJobs.length" class="job-list">
        <article v-for="job in activeJobs" :key="job.job_id">
          <div>
            <strong>{{ job.filename }}</strong>
            <span>{{ job.message }}</span>
          </div>
          <div
            class="progress-track"
            role="progressbar"
            aria-label="入库进度"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-valuenow="Math.max(...job.steps.map((step) => step.percent), 0)"
          >
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
            <small>{{ document.file_type?.replace(/^\./, "").toUpperCase() || "文档" }}</small>
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
      <div v-else class="empty-state">{{ loading ? "正在加载…" : "暂无知识文档，可上传平台规则、常见问题或售后流程。" }}</div>
    </article>
  </section>
</template>
