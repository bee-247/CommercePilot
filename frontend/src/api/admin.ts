import { apiRequest, authenticatedFetch } from "./client";

export interface ReadinessResponse {
  status: "ready" | "not_ready";
  components: Record<
    string,
    { status: "ready" | "unavailable"; reason?: string }
  >;
}

export function getHealth(): Promise<{ status: string; model: string }> {
  return apiRequest("/health");
}

export async function getReadiness(): Promise<ReadinessResponse> {
  const response = await authenticatedFetch("/health/ready");
  return response.json() as Promise<ReadinessResponse>;
}

export function getMetrics(): Promise<Record<string, unknown>> {
  return apiRequest("/api/v1/metrics", {}, true);
}

export function getExperiments(): Promise<Record<string, unknown>> {
  return apiRequest("/api/v1/experiments", {}, true);
}

export function indexProducts(): Promise<Record<string, unknown>> {
  return apiRequest("/api/v1/vector-index/products", { method: "POST" }, true);
}
