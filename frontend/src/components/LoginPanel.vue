<script setup lang="ts">
import { ref } from "vue";

import { login, register } from "../api/auth";
import { setAccessToken } from "../api/client";
import type { AuthResponse } from "../api/types";

const props = withDefaults(defineProps<{ adminOnly?: boolean }>(), {
  adminOnly: false,
});

const emit = defineEmits<{
  authenticated: [account: AuthResponse];
  unauthorized: [];
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
    if (props.adminOnly && account.role !== "admin") {
      setAccessToken("");
      emit("unauthorized");
      error.value = "该账号不是管理员，无法进入资料库";
      return;
    }
    emit("authenticated", account);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "认证失败";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <section class="auth-card" aria-labelledby="login-title">
    <div>
      <span class="product-kicker">CommercePilot 资料库</span>
      <h3 id="login-title">{{ adminOnly ? "管理员登录" : "登录资料库" }}</h3>
      <p class="muted">
        {{ adminOnly ? "仅管理员可以上传和维护平台知识文档。" : "资料库需要账户授权。" }}
      </p>
    </div>
    <form class="auth-form" @submit.prevent="submit('login')">
      <div class="auth-fields">
        <label for="login-username">
          用户名
          <input
            id="login-username"
            v-model="username"
            name="username"
            autocomplete="username"
            placeholder="输入用户名"
            required
          />
        </label>
        <label for="login-password">
          密码
          <input
            id="login-password"
            v-model="password"
            name="password"
            autocomplete="current-password"
            type="password"
            placeholder="输入密码"
            required
          />
        </label>
      </div>
      <p v-if="error" class="error-text" role="alert">{{ error }}</p>
      <div class="button-row">
        <button type="submit" class="primary" :disabled="loading">
          {{ loading ? "处理中…" : "登录" }}
        </button>
        <button
          v-if="!adminOnly"
          type="button"
          class="secondary"
          :disabled="loading"
          @click="submit('register')"
        >
          注册新账户
        </button>
      </div>
    </form>
  </section>
</template>
