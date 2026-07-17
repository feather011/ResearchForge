"""
SSE 实时推送事件
"""

import json
import asyncio
from typing import Dict, Any, AsyncGenerator


async def event_stream(research_id: str, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """
    SSE 事件流生成器

    用法:
        for event in event_stream(research_id, queue):
            yield event
    """

    # 发送连接成功事件
    yield f"event: connected\ndata: {json.dumps({'research_id': research_id})}\n\n"

    while True:
        try:
            # 等待事件（超时30秒发送心跳）
            data = await asyncio.wait_for(queue.get(), timeout=30)

            event_type = data.get("type", "message")
            payload = json.dumps(data.get("payload", {}), ensure_ascii=False)

            yield f"event: {event_type}\ndata: {payload}\n\n"

            # 如果是完成或错误事件，结束流
            if event_type in ("complete", "error"):
                break

        except asyncio.TimeoutError:
            # 发送心跳保持连接
            yield f"event: heartbeat\ndata: {{}}\n\n"


def create_progress_event(step: int, total: int, agent: str, content: str, sources: list = None) -> Dict:
    """创建进度事件"""
    payload = {
        "step": step,
        "total": total,
        "agent": agent,
        "content": content[:200]
    }
    if sources is not None:
        payload["sources"] = [
            {"id": s.get("id", f"来源{i+1}"), "title": s.get("title", "")[:100], "snippet": s.get("snippet", "")[:200]}
            for i, s in enumerate(sources[:5])
        ]
    return {"type": "progress", "payload": payload}


def create_review_event(intervention_id: str, message: str, preview: str = "") -> Dict:
    """创建审核事件"""
    return {
        "type": "review",
        "payload": {
            "intervention_id": intervention_id,
            "message": message,
            "preview": preview
        }
    }


def create_complete_event(
    report: str,
    stats: dict = None,
    claim_verification: dict = None,
    audit: dict = None,
    mode: str = "",
    duration_s: float = None,
) -> Dict:
    """创建完成事件（含元数据）"""
    payload = {"report": report}
    if stats:
        payload["stats"] = stats
    if claim_verification:
        payload["claim_verification"] = claim_verification
    if audit:
        payload["audit"] = audit
    if mode:
        payload["mode"] = mode
    if duration_s is not None:
        payload["duration_s"] = duration_s
    return {"type": "complete", "payload": payload}


def create_trace_event(trace_data: dict) -> dict:
    """创建 Trace 事件"""
    return {
        "type": "trace",
        "payload": trace_data,
    }


def create_error_event(error: str) -> Dict:  # noqa: E302 — local import only
    """创建错误事件"""
    return {
        "type": "error",
        "payload": {
            "error": error
        }
    }
