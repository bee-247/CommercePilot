from contextvars import ContextVar
from typing import Optional

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool
from core.env import load_project_env

load_project_env()

_LAST_RAG_CONTEXT: ContextVar[Optional[dict]] = ContextVar(
    "last_rag_context",
    default=None,
)
_KNOWLEDGE_TOOL_CALLS_THIS_TURN: ContextVar[int] = ContextVar(
    "knowledge_tool_calls_this_turn",
    default=0,
)
_RAG_STEP_QUEUE: ContextVar[object | None] = ContextVar(
    "rag_step_queue",
    default=None,
)
_RAG_STEP_LOOP: ContextVar[object | None] = ContextVar(
    "rag_step_loop",
    default=None,
)


def _set_last_rag_context(context: dict):
    _LAST_RAG_CONTEXT.set(context)


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    context = _LAST_RAG_CONTEXT.get()
    if clear:
        _LAST_RAG_CONTEXT.set(None)
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN.set(0)


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    _RAG_STEP_QUEUE.set(queue)
    if queue:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        _RAG_STEP_LOOP.set(loop)
    else:
        _RAG_STEP_LOOP.set(None)


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    queue = _RAG_STEP_QUEUE.get()
    loop = _RAG_STEP_LOOP.get()
    if queue is not None and loop is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, step)
        except Exception:
            pass


def _run_customer_knowledge_search(
    query: str,
    tool_name: str,
) -> str:
    call_count = _KNOWLEDGE_TOOL_CALLS_THIS_TURN.get()
    if call_count >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: knowledge search has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN.set(call_count + 1)

    from ..rewrite.rag_pipeline import run_rag_graph
    rag_result = run_rag_graph(query)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        rag_trace["tool_name"] = tool_name
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)


@tool("search_customer_knowledge")
def search_customer_knowledge(query: str) -> str:
    """Search product, policy, logistics and after-sales knowledge."""
    return _run_customer_knowledge_search(
        query,
        "search_customer_knowledge",
    )
