import { apiRequest } from "./client";
import type { DocumentInfo, DocumentJob } from "./types";

export async function listDocuments(): Promise<DocumentInfo[]> {
  const response = await apiRequest<{ documents: DocumentInfo[] }>(
    "/api/rag/documents",
    {},
    true,
  );
  return response.documents;
}

export function uploadDocument(
  file: File,
  metadata: Record<string, string>,
): Promise<{ job_id: string; filename: string; message: string }> {
  const form = new FormData();
  form.append("file", file);
  Object.entries(metadata).forEach(([key, value]) => form.append(key, value));
  return apiRequest(
    "/api/rag/documents/upload/async",
    { method: "POST", body: form },
    true,
  );
}

export function deleteDocument(
  filename: string,
): Promise<{ job_id: string; message: string }> {
  return apiRequest(
    `/api/rag/documents/delete/async/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
    true,
  );
}

export function getUploadJob(jobId: string): Promise<DocumentJob> {
  return apiRequest(
    `/api/rag/documents/upload/jobs/${encodeURIComponent(jobId)}`,
    {},
    true,
  );
}

export function getDeleteJob(jobId: string): Promise<DocumentJob> {
  return apiRequest(
    `/api/rag/documents/delete/jobs/${encodeURIComponent(jobId)}`,
    {},
    true,
  );
}
