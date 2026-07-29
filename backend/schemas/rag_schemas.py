from pydantic import BaseModel, Field
from typing import Any, Optional, List


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class CurrentUserResponse(BaseModel):
    username: str
    role: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"


class RetrievedChunk(BaseModel):
    filename: str
    resource_id: Optional[int] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    business_line: Optional[str] = None
    document_type: Optional[str] = None
    section_title: Optional[str] = None
    product_tags: Optional[List[str]] = None
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class RagTrace(BaseModel):
    tool_used: bool
    tool_name: str
    query: Optional[str] = None
    expanded_query: Optional[str] = None
    step_back_question: Optional[str] = None
    step_back_answer: Optional[str] = None
    expansion_type: Optional[str] = None
    hypothetical_doc: Optional[str] = None
    retrieval_stage: Optional[str] = None
    grade_score: Optional[str] = None
    grade_route: Optional[str] = None
    rewrite_needed: Optional[bool] = None
    rewrite_strategy: Optional[str] = None
    rewrite_query: Optional[str] = None
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    candidate_k: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    expanded_retrieved_chunks: Optional[List[RetrievedChunk]] = None


class ChatResponse(BaseModel):
    response: str
    rag_trace: Optional[RagTrace] = None
    agent_route: Optional[str] = None


class MessageInfo(BaseModel):
    type: str
    content: str
    timestamp: str
    rag_trace: Optional[RagTrace] = None


class SessionMessagesResponse(BaseModel):
    messages: List[MessageInfo]


class SessionInfo(BaseModel):
    session_id: str
    updated_at: str
    message_count: int


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]


class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str


class TokenUsageResponse(BaseModel):
    session_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0


class DocumentInfo(BaseModel):
    resource_id: Optional[int] = None
    filename: str
    display_name: Optional[str] = None
    visibility: Optional[str] = None
    is_owner: bool = False
    file_type: str
    chunk_count: int
    category: Optional[str] = None
    brand: Optional[str] = None
    business_line: Optional[str] = None
    document_type: Optional[str] = None
    status: Optional[str] = None
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class DocumentUploadResponse(BaseModel):
    resource_id: Optional[int] = None
    filename: str
    chunks_processed: int
    message: str


class DocumentUploadStartResponse(BaseModel):
    job_id: str
    resource_id: Optional[int] = None
    filename: str
    message: str


class UploadStepInfo(BaseModel):
    key: str
    label: str
    percent: int
    status: str
    message: str = ""


class DocumentUploadJobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    current_step: str
    message: str
    total_chunks: int = 0
    processed_chunks: int = 0
    error: Optional[str] = None
    created_at: str
    updated_at: str
    steps: List[UploadStepInfo]


class DocumentDeleteStartResponse(BaseModel):
    job_id: str
    filename: str
    message: str


class DocumentDeleteJobResponse(DocumentUploadJobResponse):
    pass


class DocumentDeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    message: str


class ServiceArtifactCreate(BaseModel):
    artifact_type: str = Field(
        ...,
        description="faq_set / sales_script / reply_review / service_playbook",
    )
    title: str
    prompt: str = ""
    content_json: dict = Field(default_factory=dict)
    source_chunk_ids: List[str] = Field(default_factory=list)


class ServiceArtifactUpdate(BaseModel):
    artifact_type: Optional[str] = None
    title: Optional[str] = None
    prompt: Optional[str] = None
    content_json: Optional[dict] = None
    source_chunk_ids: Optional[List[str]] = None


class ServiceArtifactInfo(BaseModel):
    id: int
    artifact_type: str
    title: str
    prompt: str
    content_json: dict
    source_chunk_ids: List[str]
    created_at: str
    updated_at: str


class ServiceArtifactListResponse(BaseModel):
    artifacts: List[ServiceArtifactInfo]


class ServiceTaskSourceChunk(BaseModel):
    chunk_id: str = ""
    filename: str = ""
    page_number: Optional[str | int] = None
    section_title: str = ""
    text: str = ""


class ServiceTaskResponse(BaseModel):
    artifact_type: str
    title: str
    content: dict
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_chunks: List[ServiceTaskSourceChunk] = Field(default_factory=list)
    verifier_notes: str = ""
    agent_route: str = ""
    saved_artifact_id: Optional[int] = None


class GenerateFaqRequest(BaseModel):
    topic: str
    category: str = ""
    brand: str = ""
    document_type: str = ""
    count: int = Field(default=5, ge=1, le=20)
    tone: str = "professional"
    save: bool = False


class GenerateSalesScriptRequest(BaseModel):
    customer_need: str
    category: str = ""
    brand: str = ""
    channel: str = "online"
    tone: str = "professional"
    save: bool = False


class ReviewServiceReplyRequest(BaseModel):
    customer_message: str
    agent_reply: str
    policy_context: str = ""
    category: str = ""
    brand: str = ""
    max_score: int = Field(default=100, ge=1, le=100)
    save: bool = False
