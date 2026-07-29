import { apiRequest, setAccessToken } from "./client";
import type { AuthResponse } from "./types";

export async function login(
  username: string,
  password: string,
): Promise<AuthResponse> {
  const response = await apiRequest<AuthResponse>("/api/rag/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setAccessToken(response.access_token);
  return response;
}

export async function register(
  username: string,
  password: string,
): Promise<AuthResponse> {
  const response = await apiRequest<AuthResponse>("/api/rag/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setAccessToken(response.access_token);
  return response;
}

export function getCurrentUser(): Promise<Pick<AuthResponse, "username" | "role">> {
  return apiRequest("/api/rag/auth/me", {}, true);
}
