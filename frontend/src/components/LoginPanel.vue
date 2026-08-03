<script setup lang="ts">
import { ref } from "vue";

import { login, register } from "../api/auth";
import type { AuthResponse } from "../api/types";

const emit = defineEmits<{
  authenticated: [account: AuthResponse];
}>();

const username = ref("");
const password = ref("");
const loading = ref(false);
const error = ref("");

async function submit(mode: "login" | "register") {
  if (!username.value.trim() || !password.value) {
    error.value = "请输入用户名和密码";
    return;
  }
  loading.value = true;
  error.value = "";
  try {
    const account =
      mode === "login"
        ? await login(username.value.trim(), password.value)
        : await register(username.value.trim(), password.value);
    emit("authenticated", account);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "认证失败";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <section class="auth-card">
    <div>
      <span class="eyebrow">KNOWLEDGE ACCOUNT</span>
      <h3>登录资料库</h3>
      <p class="muted">资料库需要账户授权。</p>
    </div>
    <div class="auth-fields">
      <input v-model="username" autocomplete="username" placeholder="用户名" />
      <input
        v-model="password"
        autocomplete="current-password"
        type="password"
        placeholder="密码"
        @keyup.enter="submit('login')"
      />
    </div>
    <p v-if="error" class="error-text">{{ error }}</p>
    <div class="button-row">
      <button class="primary" :disabled="loading" @click="submit('login')">
        {{ loading ? "处理中…" : "登录" }}
      </button>
      <button class="secondary" :disabled="loading" @click="submit('register')">
        注册新账户
      </button>
    </div>
  </section>
</template>
