import { ApiError, apiRequest, authenticatedFetch } from "./client";
import type { RagChatResponse } from "./types";

export function sendRagMessage(
  message: string,
  sessionId: string,
): Promise<RagChatResponse> {
  return apiRequest<RagChatResponse>(
    "/api/rag/chat",
    {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId }),
    },
    true,
  );
}

export async function streamRagMessage(
  message: string,
  sessionId: string,
  onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
  const response = await authenticatedFetch("/api/rag/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
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
      if (!data || data === "[DONE]") continue;
      try {
        onEvent(JSON.parse(data) as Record<string, unknown>);
      } catch {
        onEvent({ type: "content", content: data });
      }
    }
    if (done) break;
  }
}
