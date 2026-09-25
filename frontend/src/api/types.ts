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
    orchestration_mode?: string;
    orchestration_trace?: AgentMeshTrace;
    total_latency_ms: number;
  };
  token_usage: Record<string, number>;
  request_latency_ms: number;
  rag_trace: RagTrace & {
    applied?: boolean;
    error?: string;
  };
  citations: SourceCitation[];
}

export interface SalesChatSession {
  session_id: string;
  updated_at: string;
  message_count: number;
  preview: string;
}

export interface AgentMeshTaskTrace {
  task_id: string;
  capability: string;
  depends_on: string[];
  agent_id: string;
  broker_score: number;
  broker_mode?: string;
  broker_fallback_reason?: string | null;
  broker_reason?: string;
  strategy?: { mode: string; reason?: string };
  agent_bid?: {
    can_handle: boolean;
    scenario_score: number;
    reason: string;
  };
  business_success?: boolean | null;
  quality_feedback_source?: string | null;
  broker_candidates?: Array<{
    agent_id: string;
    score: number;
    selected: boolean;
    active_tasks: number;
    components: {
      historical_success: number;
      latency: number;
      cost: number;
      scenario: number;
    };
    bid: {
      can_handle: boolean;
      scenario_score: number;
      reason: string;
    };
    rule_bid?: {
      can_handle: boolean;
      scenario_score: number;
      reason: string;
    };
    llm_assessment?: {
      agent_id: string;
      can_handle: boolean;
      scenario_score: number;
      reason: string;
    } | null;
    routing_mode?: string;
    routing_fallback_reason?: string | null;
  }>;
  status: "running" | "completed" | "failed";
  latency_ms: number;
  output_keys: string[];
  error?: string;
}

export interface AgentMeshTrace {
  architecture: string;
  agent_config_version?: number;
  planner: string;
  broker: string;
  blackboard: string;
  plans: Array<{
    phase: string;
    reason: string;
    planner?: string;
    fallback_reason?: string;
    tasks: Array<{
      task_id: string;
      capability: string;
      depends_on: string[];
    }>;
  }>;
  waves: string[][];
  tasks: AgentMeshTaskTrace[];
  replan: {
    triggered: boolean;
    reason: string;
    added_tasks: string[];
    planner?: string;
    fallback_reason?: string;
  };
  blackboard_keys: string[];
  completed_task_ids: string[];
  broker_weights?: Record<string, number>;
  broker_runtime_policy?: {
    success_prior_calls: number;
    latency_ceiling_ms: number;
  };
  evaluation?: {
    initial_judge_passed: boolean | null;
    final_judge_passed: boolean | null;
    business_success: boolean;
    replan_recovered: boolean;
    persisted: boolean;
  };
  total_latency_ms?: number;
  fallback?: string;
  error?: string;
}

export interface SourceCitation {
  filename: string;
  page_number?: string | number;
  section_title?: string;
  chunk_id?: string;
  text?: string;
  target_product_id?: string;
  target_product_name?: string;
}

export interface RagTrace {
  query?: string;
  retrieval_stage?: string;
  retrieval_mode?: string;
  retrieval_modes?: string[];
  retrieval_profile?: string;
  product_query_count?: number;
  parallel_retrieval?: boolean;
  evidence_queries?: Array<{
    product_id: string;
    product_name: string;
    query: string;
  }>;
  rewrite_strategy?: string;
  rerank_applied?: boolean;
  auto_merge_applied?: boolean;
  retrieved_chunks?: Array<{
    filename: string;
    page_number?: string | number;
    section_title?: string;
    text?: string;
    target_product_id?: string;
    target_product_name?: string;
    evidence_query?: string;
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
