import { apiRequest } from "./client";
import type { RagSession, RagTrace } from "./types";

export async function listSessions(): Promise<RagSession[]> {
  const response = await apiRequest<{ sessions: RagSession[] }>(
    "/api/rag/sessions",
    {},
    true,
  );
  return response.sessions;
}

export async function getSessionMessages(
  sessionId: string,
): Promise<Array<{ type: string; content: string; rag_trace?: RagTrace }>> {
  const response = await apiRequest<{
    messages: Array<{ type: string; content: string; rag_trace?: RagTrace }>;
  }>(`/api/rag/sessions/${encodeURIComponent(sessionId)}`, {}, true);
  return response.messages;
}

export function deleteSession(sessionId: string): Promise<{ message: string }> {
  return apiRequest(
    `/api/rag/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE" },
    true,
  );
}
