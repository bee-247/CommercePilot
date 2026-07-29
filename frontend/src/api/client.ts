const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const TOKEN_KEY = "commerce-pilot-token";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export function getAccessToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setAccessToken(token: string): void {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
  authenticated = false,
): Promise<T> {
  const headers = new Headers(init.headers);
  if (!(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (authenticated) {
    const token = getAccessToken();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = await response.json();
      message = payload.detail ?? payload.message ?? message;
    } catch {
      // Keep the status-based fallback for non-JSON responses.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export async function authenticatedFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getAccessToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return fetch(`${API_BASE_URL}${path}`, { ...init, headers });
}
