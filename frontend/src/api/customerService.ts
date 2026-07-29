import { apiRequest } from "./client";
import type { ServiceTaskResponse } from "./types";


export function generateFaq(payload: {
  topic: string;
  category: string;
  brand: string;
  count: number;
  tone: string;
}): Promise<ServiceTaskResponse> {
  return apiRequest(
    "/api/rag/customer-service/faq/generate",
    {
      method: "POST",
      body: JSON.stringify({ ...payload, document_type: "", save: false }),
    },
    true,
  );
}


export function generateSalesScript(payload: {
  customer_need: string;
  category: string;
  brand: string;
  channel: string;
  tone: string;
}): Promise<ServiceTaskResponse> {
  return apiRequest(
    "/api/rag/customer-service/scripts/generate",
    {
      method: "POST",
      body: JSON.stringify({ ...payload, save: false }),
    },
    true,
  );
}


export function reviewServiceReply(payload: {
  customer_message: string;
  agent_reply: string;
  policy_context: string;
  category: string;
  brand: string;
}): Promise<ServiceTaskResponse> {
  return apiRequest(
    "/api/rag/customer-service/replies/review",
    {
      method: "POST",
      body: JSON.stringify({ ...payload, max_score: 100, save: false }),
    },
    true,
  );
}
