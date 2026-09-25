<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import { getCurrentUser } from "./api/auth";
import { getAccessToken, setAccessToken } from "./api/client";
import type { AuthResponse } from "./api/types";
import LoginPanel from "./components/LoginPanel.vue";
import ChatPage from "./pages/ChatPage.vue";
import DocumentsPage from "./pages/DocumentsPage.vue";
import "./styles/app.css";

type PageKey = "chat" | "documents";

const currentPage = ref<PageKey>("chat");
const account = ref<AuthResponse | null>(null);
const hasToken = ref(Boolean(getAccessToken()));
const isAdmin = computed(() => account.value?.role === "admin");
const activePageLabel = computed(() =>
  currentPage.value === "documents" ? "资料库管理" : "智导",
);
const activePageTagline = computed(() =>
  currentPage.value === "documents"
    ? "维护平台客服回答所需的通用知识文档"
    : "更懂需求的智能购物助手",
);

function authenticated(value: AuthResponse) {
  account.value = value;
  hasToken.value = true;
}

function clearAuthentication() {
  setAccessToken("");
  account.value = null;
  hasToken.value = false;
}

function logout() {
  clearAuthentication();
  currentPage.value = "chat";
}

function openDocuments() {
  currentPage.value = "documents";
}

function openChat() {
  currentPage.value = "chat";
}

onMounted(async () => {
  if (!hasToken.value) return;
  try {
    const current = await getCurrentUser();
    account.value = {
      ...current,
      access_token: getAccessToken(),
    };
  } catch {
    logout();
  }
});
</script>

<template>
  <div class="app-shell">
    <main class="main">
      <header class="topbar">
        <div class="topbar-brand">
          <div class="brand-mark" aria-hidden="true">智</div>
          <div>
            <span class="product-kicker">CommercePilot</span>
            <h1>{{ activePageLabel }}</h1>
            <p class="topbar-tagline">{{ activePageTagline }}</p>
          </div>
        </div>
        <div class="topbar-center" aria-label="工作台状态">
          <div class="topbar-signal" aria-hidden="true">
            <i></i><i></i><i></i><i></i><i></i>
          </div>
          <div class="topbar-center-copy">
            <strong>{{ currentPage === "chat" ? "智能导购空间" : "知识资料中枢" }}</strong>
            <span>
              {{ currentPage === "chat" ? "商品库与推荐引擎已连接" : "资料索引服务已就绪" }}
            </span>
          </div>
          <span class="topbar-live"><b></b>在线</span>
        </div>
        <div class="topbar-actions">
          <div v-if="currentPage === 'chat'" class="mode-indicator">
            混合检索模式
          </div>
          <button
            v-if="currentPage === 'chat'"
            type="button"
            class="admin-entry-button"
            @click="openDocuments"
          >
            资料库
          </button>
          <template v-else>
            <span v-if="isAdmin" class="admin-identity">
              管理员 / {{ account?.username }}
            </span>
            <button type="button" class="admin-entry-button" @click="openChat">
              返回导购
            </button>
            <button
              v-if="isAdmin"
              type="button"
              class="text-button"
              @click="logout"
            >
              退出登录
            </button>
          </template>
        </div>
      </header>

      <LoginPanel
        v-if="currentPage === 'documents' && !isAdmin"
        admin-only
        @authenticated="authenticated"
        @unauthorized="clearAuthentication"
      />
      <template v-else>
        <!-- 导购是公开访客空间，不能随资料库管理员登录切换会话命名空间。 -->
        <ChatPage v-if="currentPage === 'chat'" :user-id="'web_user'" />
        <DocumentsPage v-else />
      </template>
    </main>
  </div>
</template>
