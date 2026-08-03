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

const allPages: Array<{ key: PageKey; label: string; note: string }> = [
  { key: "chat", label: "导购对话", note: "Sales" },
  { key: "documents", label: "资料库", note: "Documents" },
];

const currentPage = ref<PageKey>("chat");
const account = ref<AuthResponse | null>(null);
const hasToken = ref(Boolean(getAccessToken()));
const requiresAuth = computed(() => currentPage.value !== "chat");
const activePage = computed(
  () =>
    allPages.find((page) => page.key === currentPage.value) ?? allPages[0],
);

function authenticated(value: AuthResponse) {
  account.value = value;
  hasToken.value = true;
}

function logout() {
  setAccessToken("");
  account.value = null;
  hasToken.value = false;
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
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">智</div>
        <div>
          <h1>智导</h1>
          <p>CommercePilot</p>
        </div>
      </div>

      <nav class="nav" aria-label="工作台导航">
        <button
          v-for="page in allPages"
          :key="page.key"
          class="nav-item"
          :class="{ active: currentPage === page.key }"
          @click="currentPage = page.key"
        >
          <span>{{ page.label }}</span>
          <small>{{ page.note }}</small>
        </button>
      </nav>

      <div class="sidebar-card">
        <span class="label">当前账户</span>
        <strong>{{ account?.username || (hasToken ? "已保存凭证" : "访客") }}</strong>
        <button v-if="hasToken" class="text-button" @click="logout">退出登录</button>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <span class="eyebrow">COMMERCEPILOT WORKSPACE</span>
          <h1>{{ activePage.label }}</h1>
        </div>
        <div class="mode-indicator">
          <span></span>
          混合推荐与知识检索
        </div>
      </header>

      <LoginPanel
        v-if="requiresAuth && !hasToken"
        @authenticated="authenticated"
      />
      <template v-else>
        <ChatPage
          v-if="currentPage === 'chat'"
          :user-id="account?.username || 'web_user'"
        />
        <DocumentsPage v-else />
      </template>
    </main>
  </div>
</template>
