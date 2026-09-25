import { ApiError, apiRequest, authenticatedFetch } from "./client";
import type { SalesChatResponse, SalesChatSession } from "./types";

export interface SalesProgressEvent {
  type: "progress";
  stage: string;
  label: string;
  status: "running" | "completed" | "failed";
  elapsed_ms: number;
  latency_ms?: number;
  agent_id?: string;
  depends_on?: string[];
  error?: string | null;
}

export function sendSalesMessage(
  message: string,
  sessionId: string,
  userId: string,
): Promise<SalesChatResponse> {
  return apiRequest<SalesChatResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      session_id: sessionId,
      user_id: userId,
      num_items: 6,
    }),
  });
}

export async function streamSalesMessage(
  message: string,
  sessionId: string,
  userId: string,
  onProgress: (event: SalesProgressEvent) => void,
): Promise<SalesChatResponse> {
  const response = await authenticatedFetch("/api/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      user_id: userId,
      num_items: 6,
    }),
  });
  if (!response.ok) {
    throw new ApiError(`流式请求失败（${response.status}）`, response.status);
  }
  if (!response.body) {
    throw new ApiError("浏览器未提供流式响应", 500);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: SalesChatResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("");
      if (!data) continue;
      const event = JSON.parse(data) as Record<string, unknown>;
      if (event.type === "progress") {
        onProgress(event as unknown as SalesProgressEvent);
      } else if (event.type === "result") {
        result = event.data as SalesChatResponse;
      } else if (event.type === "error") {
        throw new ApiError(String(event.content || "Agent 执行失败"), 500);
      }
    }
    if (done) break;
  }

  if (!result) {
    throw new ApiError("流式响应未返回最终结果", 500);
  }
  return result;
}

export function sendSalesImage(
  image: File,
  message: string,
  sessionId: string,
  userId: string,
): Promise<SalesChatResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("message", message);
  form.append("session_id", sessionId);
  form.append("user_id", userId);
  return apiRequest<SalesChatResponse>("/api/v1/chat/image", {
    method: "POST",
    body: form,
  });
}

export async function listSalesChatSessions(
  userId: string,
): Promise<SalesChatSession[]> {
  const response = await apiRequest<{ sessions: SalesChatSession[] }>(
    `/api/v1/chat/sessions?user_id=${encodeURIComponent(userId)}`,
  );
  return response.sessions;
}

export async function getSalesChatSessionMessages(
  userId: string,
  sessionId: string,
): Promise<Array<{ role: "user" | "assistant"; content: string }>> {
  const response = await apiRequest<{
    messages: Array<{ role: "user" | "assistant"; content: string }>;
  }>(
    `/api/v1/chat/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
  );
  return response.messages;
}

export async function deleteSalesChatSession(
  userId: string,
  sessionId: string,
): Promise<boolean> {
  const response = await apiRequest<{ deleted: boolean }>(
    `/api/v1/chat/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" },
  );
  return response.deleted;
}
