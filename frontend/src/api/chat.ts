import { apiRequest } from "./client";
import type { SalesChatResponse } from "./types";

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
