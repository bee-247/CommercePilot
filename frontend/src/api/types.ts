export interface Product {
  product_id: string;
  name: string;
  category: string;
  price: number;
  brand: string;
  stock: number;
  description: string;
}

export interface SalesChatResponse {
  answer: string;
  recall_strategy: string;
  recall_reason: string;
  recommendation: {
    request_id: string;
    products: Product[];
    marketing_copies: Array<{ product_id: string; copy: string }>;
    experiment_group: string;
    total_latency_ms: number;
  };
  token_usage: Record<string, number>;
  rag_trace: RagTrace & {
    applied?: boolean;
    error?: string;
  };
  citations: SourceCitation[];
}

export interface SourceCitation {
  filename: string;
  page_number?: string | number;
  section_title?: string;
  chunk_id?: string;
  text?: string;
}

export interface RagTrace {
  query?: string;
  retrieval_stage?: string;
  retrieval_mode?: string;
  rewrite_strategy?: string;
  rerank_applied?: boolean;
  auto_merge_applied?: boolean;
  retrieved_chunks?: Array<{
    filename: string;
    page_number?: string | number;
    section_title?: string;
    text?: string;
  }>;
}

export interface RagChatResponse {
  response: string;
  rag_trace?: RagTrace;
  agent_route?: string;
}

export interface DocumentInfo {
  resource_id?: number;
  filename: string;
  display_name?: string;
  visibility?: string;
  is_owner: boolean;
  file_type: string;
  chunk_count: number;
  category?: string;
  brand?: string;
  business_line?: string;
  document_type?: string;
  status?: string;
}

export interface DocumentJob {
  job_id: string;
  filename: string;
  status: "pending" | "running" | "completed" | "failed";
  current_step: string;
  message: string;
  total_chunks: number;
  processed_chunks: number;
  error?: string;
  steps: Array<{
    key: string;
    label: string;
    percent: number;
    status: string;
    message: string;
  }>;
}

export interface AuthResponse {
  access_token: string;
  username: string;
  role: string;
}

export interface ServiceTaskResponse {
  artifact_type: string;
  title: string;
  content: Record<string, unknown>;
  source_chunk_ids: string[];
  verifier_notes: string;
  agent_route: string;
  source_chunks: Array<{
    chunk_id: string;
    filename: string;
    page_number?: string | number;
    section_title: string;
    text: string;
  }>;
  saved_artifact_id?: number;
}

export interface RagSession {
  session_id: string;
  updated_at: string;
  message_count: number;
}
