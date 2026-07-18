"""
FastAPI 应用入口（新：使用 ResearchService）
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import asyncio
import uuid
import logging
import threading
import datetime
from typing import Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .models import ResearchRequest, ReviewRequest, ResearchResponse, StatusResponse, ReviewResponse
from .limiter import RateLimiter
from .sse import event_stream, create_progress_event, create_review_event, create_complete_event, create_error_event, create_trace_event
from .persist import save_task, load_task, list_tasks
from .config import settings
from ..orchestration import ResearchMode
from ..orchestration.checkpoint_store import CheckpointStore
from ..trace import TraceCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI")

app = FastAPI(title="ResearchForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 全局存储
research_tasks: Dict[str, dict] = {}
limiter = RateLimiter(max_requests=settings.RATE_LIMIT_MAX, window_seconds=settings.RATE_LIMIT_WINDOW)
sse_queues: Dict[str, asyncio.Queue] = {}
checkpoint_store = CheckpointStore()


def _record_event(task: dict, event_type: str, payload: dict):
    """记录事件到 task 的历史日志"""
    events = task.setdefault("events", [])
    events.append({
        "type": event_type,
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.datetime.utcnow().microsecond // 1000:03d}Z",
        **payload
    })

# 启动时从文件加载已完成的研究
import json as _json
data_reload_count = 0
for _f in sorted(Path(__file__).parent.glob("data/*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
    try:
        with open(_f, "r", encoding="utf-8") as _fp:
            _rec = _json.load(_fp)
        _rid = _rec.get("research_id", _f.stem)
        if _rid not in research_tasks:
            research_tasks[_rid] = _rec
            data_reload_count += 1
    except:
        pass
if data_reload_count:
    logger.info(f"从文件加载了 {data_reload_count} 条历史研究记录")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": "ResearchForge API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/research": "启动研究",
            "GET  /api/status/{id}": "查询状态",
            "GET  /api/stream/{id}": "SSE实时推送",
            "POST /api/review/{id}": "人工审核",
            "POST /api/research/{id}/resume": "恢复研究",
            "POST /api/sse-test": "SSE快速模拟测试",
        }
    }


def run_research_sync(research_id: str, topic: str, llm, mode: ResearchMode):
    """在线程中同步执行研究（新 ResearchService）"""
    from researchforge.research_service import ResearchService

    task = research_tasks[research_id]
    queue = sse_queues.get(research_id)

    # 创建 TraceCollector，日志和持久化共用
    tracer = TraceCollector(run_id=research_id)

    try:
        task["state"] = "initializing"
        if queue:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            evt = create_progress_event(0, 10, "System", "正在初始化...")
            loop.run_until_complete(queue.put(evt))
            _record_event(task, "progress", evt["payload"])

        def progress_callback(agent_name, message, extra_data=None):
            if queue:
                try:
                    if extra_data:
                        evt = create_progress_event(1, 10, agent_name, message, sources=extra_data.get("sources"))
                    else:
                        evt = create_progress_event(1, 10, agent_name, message)
                    loop.run_until_complete(queue.put(evt))
                    _record_event(task, "progress", evt["payload"])
                except:
                    pass

        # Trace 回调：SSE 推送 + 持久化
        def trace_callback(trace_event):
            if not queue:
                return
            try:
                evt = create_trace_event(trace_event.to_dict())
                loop.run_until_complete(queue.put(evt))
                _record_event(task, "trace", evt["payload"])
            except:
                pass

        tracer.callback = trace_callback

        # 心跳
        def heartbeat_push():
            import time as t2
            while task.get("state") in ("running", "initializing"):
                t2.sleep(15)
                if queue:
                    try:
                        evt = create_progress_event(1, 10, "System", f"处理中...")
                        loop.run_until_complete(queue.put(evt))
                    except:
                        pass

        hb_thread = threading.Thread(target=heartbeat_push)
        hb_thread.daemon = True
        hb_thread.start()

        task["state"] = "running"

        # 使用 ResearchService 执行
        svc = ResearchService(llm=llm, checkpoint_store=checkpoint_store)
        result = svc.run(topic, mode=mode, progress_callback=progress_callback, tracer=tracer, task_id=research_id)

        # 检查是否需要人工审核
        if result.get("require_human_review"):
            import uuid as _uuid
            review_id = _uuid.uuid4().hex[:8]
            task["state"] = "awaiting_review"
            task["intervention_id"] = review_id
            task["report"] = result.get("report", "")

            if queue:
                preview = result.get("report", "")[:1000]
                evt = create_review_event(review_id, "研究报告已完成，请审核", preview)
                loop.run_until_complete(queue.put(evt))
                _record_event(task, "review", evt["payload"])
            save_task(research_id, task)
            return  # 等待审核，不继续推送 complete

        task["state"] = "completed"
        task["report"] = result["report"]

        if queue:
            evt = create_complete_event(
                report=result["report"],
                stats=result.get("stats", {}),
                claim_verification=result.get("claim_verification", {}),
                audit=result.get("audit", {}),
                mode=result.get("mode", mode.value),
                duration_s=result.get("_duration_s"),
            )
            loop.run_until_complete(queue.put(evt))
            _record_event(task, "complete", evt["payload"])
        save_task(research_id, task)
        sse_queues.pop(research_id, None)

    except Exception as e:
        task["state"] = "failed"
        logger.error(f"研究失败: {e}")
        save_task(research_id, task)
        if queue:
            try:
                evt = {"type": "error", "payload": {"error": str(e)}}
                loop.run_until_complete(queue.put(evt))
            except:
                pass


@app.post("/api/research")
async def start_research(req: ResearchRequest):
    if not limiter.check("research"):
        raise HTTPException(429, detail="请求太频繁")

    research_id = str(uuid.uuid4())[:8]

    from researchforge.core import BailianProvider, OllamaProvider

    if settings.LLM_PROVIDER == "ollama":
        try:
            llm = OllamaProvider(model=settings.MODEL, base_url=settings.OLLAMA_BASE_URL, timeout=settings.LLM_TIMEOUT)
        except Exception as e:
            raise HTTPException(500, detail=f"Ollama初始化失败: {e}")
    else:
        try:
            llm = BailianProvider(model=settings.MODEL, timeout=settings.LLM_TIMEOUT)
        except Exception as e:
            raise HTTPException(500, detail=f"LLM初始化失败: {e}")

    research_tasks[research_id] = {
        "topic": req.topic, "state": "pending",
        "mode": req.mode, "report": None,
    }
    sse_queues[research_id] = asyncio.Queue()

    try:
        mode = ResearchMode(req.mode.lower())
    except ValueError:
        raise HTTPException(400, detail=f"无效模式: {req.mode}，可选 fast/standard/deep")

    # 在线程中执行（不阻塞API）
    thread = threading.Thread(
        target=run_research_sync,
        args=(research_id, req.topic, llm, mode),
    )
    thread.daemon = True
    thread.start()

    logger.info(f"启动研究: {research_id} - {req.topic} ({req.mode})")

    _cleanup_stale_tasks()

    return ResearchResponse(research_id=research_id, topic=req.topic, state="pending", message=f"研究已启动（{req.mode}模式）")


@app.get("/api/status/{research_id}")
async def get_status(research_id: str):
    task = research_tasks.get(research_id)
    if not task:
        raise HTTPException(404, detail="研究不存在")
    return {
        "research_id": research_id,
        "state": task["state"],
        "topic": task["topic"],
        "intervention_id": task.get("intervention_id"),
        "report": task.get("report")
    }


@app.get("/api/stream/{research_id}")
async def stream_events(research_id: str):
    if research_id not in research_tasks:
        raise HTTPException(404, detail="研究不存在")
    queue = sse_queues.get(research_id)
    if not queue:
        raise HTTPException(404, detail="SSE队列不存在")
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        event_stream(research_id, queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/api/review/{research_id}")
async def review_research(research_id: str, req: ReviewRequest):
    task = research_tasks.get(research_id)
    if not task:
        raise HTTPException(404, detail="研究不存在")
    if task["state"] != "awaiting_review":
        raise HTTPException(400, detail="当前状态不需要审核")

    iv = task.get("intervention_id")
    queue = sse_queues.get(research_id)

    if req.response == "通过":
        task["state"] = "completed"
        task["report"] = task.get("report") or "审核通过"

        if queue:
            evt = create_complete_event(report=task["report"])
            await queue.put(evt)
            _record_event(task, "complete", evt["payload"])
        save_task(research_id, task)
        return ReviewResponse(state="completed", message="已通过，研究完成", report=task["report"])

    elif req.response == "驳回":
        task["state"] = "rejected"
        if queue:
            evt = create_complete_event(report="")
            await queue.put(evt)
            _record_event(task, "complete", evt["payload"])
        save_task(research_id, task)
        return ReviewResponse(state="rejected", message=f"已驳回: {req.comments or '无意见'}")

    elif req.response == "修改意见":
        # 保留审核状态，记录意见到 review_comments
        task["review_comments"] = req.comments or "请修改"
        if queue:
            evt = create_progress_event(1, 10, "System",
                f"审核意见: {req.comments or '请修改'}")
            await queue.put(evt)
            _record_event(task, "progress", evt["payload"])
        save_task(research_id, task)
        return ReviewResponse(state="awaiting_review",
            message=f"已记录修改意见: {req.comments or '请修改'}")

    else:
        raise HTTPException(400, detail="无效操作，可选: 通过 / 驳回 / 修改意见")


@app.post("/api/sse-test")
async def sse_test():
    """快速SSE测试端点 - 模拟完整事件周期"""
    research_id = f"test_{uuid.uuid4().hex[:6]}"

    research_tasks[research_id] = {
        "topic": "[SSE测试]模拟研究", "state": "pending",
        "agent": None, "intervention_id": None, "report": None
    }
    sse_queues[research_id] = asyncio.Queue()

    async def simulate():
        q = sse_queues[research_id]
        t = research_tasks[research_id]

        evt1 = create_progress_event(1, 4, "Planner", "正在制定研究计划...")
        await q.put(evt1)
        _record_event(t, "progress", evt1["payload"])
        t["state"] = "running"

        await asyncio.sleep(1)
        evt2 = create_progress_event(2, 4, "SearchAgent", "搜索相关资料...")
        await q.put(evt2)
        _record_event(t, "progress", evt2["payload"])

        await asyncio.sleep(1)
        evt3 = create_progress_event(3, 4, "AnalystAgent", "分析数据中...")
        await q.put(evt3)
        _record_event(t, "progress", evt3["payload"])

        await asyncio.sleep(1)
        evt4 = create_progress_event(4, 4, "WriterAgent", "撰写报告...")
        await q.put(evt4)
        _record_event(t, "progress", evt4["payload"])

        await asyncio.sleep(1)
        iv = f"test_iv_{uuid.uuid4().hex[:4]}"
        t["intervention_id"] = iv
        t["state"] = "awaiting_review"
        preview_content = "【研究计划】\n  1. 搜索AI Agent相关资料\n  2. 分析ReAct模式原理\n  3. 撰写总结报告\n\n【阶段性结果】\n  [1] AI Agent是一种能够自主感知环境、做出决策并执行动作的智能体。ReAct模式将推理(Reasoning)和行动(Action)结合，让Agent在思考中指导行动，在行动中验证思考。\n  [2] 核心思想：Think → Act → Observe 循环。"
        evt5 = create_review_event(iv, "请审核研究报告", preview_content)
        await q.put(evt5)
        _record_event(t, "review", evt5["payload"])

        waited = 0
        while t["state"] == "awaiting_review" and waited < 300:
            await asyncio.sleep(1)
            waited += 1

        if t["state"] == "awaiting_review":
            t["state"] = "failed"
            evt6 = {"type": "error", "payload": {"error": "审核超时"}}
            await q.put(evt6)
            _record_event(t, "error", evt6["payload"])
        sse_queues.pop(research_id, None)

    asyncio.create_task(simulate())

    logger.info(f"SSE测试启动: {research_id}")
    return ResearchResponse(research_id=research_id, topic="[SSE测试]模拟研究", state="running", message="SSE模拟测试已启动")


@app.get("/api/checkpoints")
async def list_recoverable_checkpoints():
    """列出可恢复的检查点（failed 状态的）"""
    ids = checkpoint_store.list_ids()
    recoverable = []
    for rid in ids[-20:]:  # 最近 20 个
        state = checkpoint_store.load(rid)
        if state and state.status == "failed":
            recoverable.append({
                "task_id": rid,
                "topic": state.topic,
                "mode": state.mode.value if hasattr(state.mode, "value") else str(state.mode),
                "failed_node": state.failed_node or "",
                "completed_nodes": list(state.completed_nodes or []),
            })
    return {"checkpoints": recoverable}


@app.get("/api/history")
async def get_history():
    """获取历史研究列表"""
    tasks = list_tasks(20)
    # 也合并内存中的活跃任务
    for rid, task in research_tasks.items():
        if task.get("state") in ("running", "awaiting_review", "pending"):
            tasks.insert(0, {
                "research_id": rid,
                "topic": task.get("topic", ""),
                "state": task.get("state", ""),
                "mode": task.get("mode", ""),
                "time": 0,
            })
    # 对 file-persisted 任务也补充 mode（已有 research_id）
    for t in tasks:
        rid = t.get("research_id", t.get("id", ""))
        if not t.get("mode") and rid:
            # 从检查点读取 mode
            state = checkpoint_store.load(rid)
            if state:
                mode_val = state.mode
                if hasattr(mode_val, "value"):
                    mode_val = mode_val.value
                t["mode"] = str(mode_val)
    return {"tasks": tasks}


@app.get("/api/research/{research_id}/events")
async def get_research_events(research_id: str):
    """获取研究事件历史"""
    # 先查内存
    task = research_tasks.get(research_id)
    if task:
        return {"research_id": research_id, "events": task.get("events", [])}
    # 再查文件持久化
    record = load_task(research_id)
    if not record:
        raise HTTPException(404, detail="研究不存在")
    return {"research_id": research_id, "events": record.get("events", [])}


@app.delete("/api/research/{research_id}")
async def delete_research(research_id: str):
    """删除研究（内存 + 文件）"""
    # 从内存删除
    research_tasks.pop(research_id, None)
    sse_queues.pop(research_id, None)
    # 从文件删除
    fpath = Path(__file__).parent / "data" / f"{research_id}.json"
    if fpath.exists():
        fpath.unlink()
    return {"status": "deleted", "research_id": research_id}


@app.post("/api/research/{research_id}/resume")
async def resume_research(research_id: str):
    """恢复一项已中断的研究"""
    from researchforge.research_service import ResearchService

    # 检查检查点是否存在
    state = checkpoint_store.load(research_id)
    if state is None:
        raise HTTPException(404, detail=f"研究 {research_id} 不存在或已损坏，无法恢复")

    if state.status == "completed":
        raise HTTPException(400, detail=f"研究已完成，无需恢复")

    # 初始化或更新内存记录
    research_tasks[research_id] = {
        "topic": state.topic,
        "state": "pending",
        "mode": state.mode.value if hasattr(state.mode, "value") else state.mode,
        "report": None,
    }
    if research_id not in sse_queues:
        sse_queues[research_id] = asyncio.Queue()

    from researchforge.core import BailianProvider, OllamaProvider
    if settings.LLM_PROVIDER == "ollama":
        llm = OllamaProvider(model=settings.MODEL, base_url=settings.OLLAMA_BASE_URL, timeout=settings.LLM_TIMEOUT)
    else:
        llm = BailianProvider(model=settings.MODEL, timeout=settings.LLM_TIMEOUT)

    thread = threading.Thread(
        target=_run_resume_sync,
        args=(research_id, llm),
    )
    thread.daemon = True
    thread.start()

    logger.info(f"恢复研究: {research_id}")
    return ResearchResponse(
        research_id=research_id,
        topic=state.topic,
        state="pending",
        message="研究已开始恢复",
    )


def _run_resume_sync(research_id: str, llm):
    """在线程中执行恢复"""
    from researchforge.research_service import ResearchService

    task = research_tasks.get(research_id, {})
    queue = sse_queues.get(research_id)
    tracer = TraceCollector(run_id=research_id)

    try:
        task["state"] = "initializing"
        if queue:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            evt = create_progress_event(0, 10, "System", "正在恢复研究...")
            loop.run_until_complete(queue.put(evt))
            _record_event(task, "progress", evt["payload"])

        def progress_callback(agent_name, message, extra_data=None):
            if queue:
                try:
                    if extra_data:
                        evt = create_progress_event(1, 10, agent_name, message, sources=extra_data.get("sources"))
                    else:
                        evt = create_progress_event(1, 10, agent_name, message)
                    loop.run_until_complete(queue.put(evt))
                    _record_event(task, "progress", evt["payload"])
                except:
                    pass

        def trace_callback(trace_event):
            if not queue:
                return
            try:
                evt = create_trace_event(trace_event.to_dict())
                loop.run_until_complete(queue.put(evt))
                _record_event(task, "trace", evt["payload"])
            except:
                pass

        tracer.callback = trace_callback

        def heartbeat_push():
            import time as t2
            while task.get("state") in ("running", "initializing"):
                t2.sleep(15)
                if queue:
                    try:
                        evt = create_progress_event(1, 10, "System", "处理中...")
                        loop.run_until_complete(queue.put(evt))
                    except:
                        pass

        hb_thread = threading.Thread(target=heartbeat_push)
        hb_thread.daemon = True
        hb_thread.start()

        task["state"] = "running"

        svc = ResearchService(llm=llm, checkpoint_store=checkpoint_store)
        result = svc.resume(research_id, progress_callback=progress_callback, tracer=tracer)

        if result.get("require_human_review"):
            import uuid as _uuid
            review_id = _uuid.uuid4().hex[:8]
            task["state"] = "awaiting_review"
            task["intervention_id"] = review_id
            task["report"] = result.get("report", "")

            if queue:
                preview = result.get("report", "")[:1000]
                evt = create_review_event(review_id, "研究报告已完成，请审核", preview)
                loop.run_until_complete(queue.put(evt))
                _record_event(task, "review", evt["payload"])
            save_task(research_id, task)
            return

        task["state"] = "completed"
        task["report"] = result["report"]

        if queue:
            evt = create_complete_event(
                report=result["report"],
                stats=result.get("stats", {}),
                claim_verification=result.get("claim_verification", {}),
                audit=result.get("audit", {}),
                mode=result.get("mode", ""),
                duration_s=result.get("_duration_s"),
            )
            loop.run_until_complete(queue.put(evt))
            _record_event(task, "complete", evt["payload"])
        save_task(research_id, task)
        sse_queues.pop(research_id, None)

    except Exception as e:
        task["state"] = "failed"
        logger.error(f"恢复失败: {e}")
        save_task(research_id, task)
        if queue:
            try:
                evt = {"type": "error", "payload": {"error": str(e)}}
                loop.run_until_complete(queue.put(evt))
            except:
                pass


def _cleanup_stale_tasks():
    """清理内存中过期的研究任务（超过 100 条的 terminal 状态）"""
    terminal_states = ("completed", "failed", "rejected")
    stale_ids = [
        rid for rid, task in research_tasks.items()
        if task.get("state") in terminal_states
    ]
    if len(stale_ids) > 50:  # 只清理超过 50 个 terminal 任务时
        keep = set(stale_ids[:-40])  # 保留最近 40 个
        for rid in keep:
            research_tasks.pop(rid, None)
