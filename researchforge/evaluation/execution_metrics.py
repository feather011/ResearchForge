"""
ExecutionMetrics — Trace 聚合执行指标

从已有 Trace 事件中聚合执行统计数据，不新增任何业务逻辑。
"""

from typing import Any, Dict, List, Optional


def build_execution_metrics(
    traces: List[dict],
    duration_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    从 Trace 事件聚合执行指标。

    Args:
        traces: 全部 trace 事件列表（tracer.get_all() 格式）
        duration_s: 可选，覆盖 trace 计算的总耗时

    Returns:
        execution metrics dict
    """
    if not traces:
        return {
            "duration_s": round(duration_s, 2) if duration_s is not None else 0.0,
            "retry_count": 0,
            "retry_exhausted_count": 0,
            "degraded_count": 0,
            "resume_count": 0,
            "node_durations": {},
            "slowest_node": "",
        }

    # ── 统计事件计数 ──
    retry_count = 0
    retry_exhausted_count = 0
    degraded_count = 0
    resume_count = 0

    for t in traces:
        stage = t.get("stage", "")
        if stage == "retry":
            retry_count += 1
        elif stage == "retry_exhausted":
            retry_exhausted_count += 1
        elif stage == "degraded":
            degraded_count += 1
        elif stage == "resume_started":
            resume_count += 1

    # ── 总耗时：从 timestamp 计算或使用传入值 ──
    if duration_s is None:
        timestamps = [t.get("timestamp", 0) for t in traces if t.get("timestamp")]
        if timestamps:
            _dur = round(max(timestamps) - min(timestamps), 2)
        else:
            _dur = 0.0
    else:
        _dur = round(duration_s, 2)

    # ── 节点耗时聚合 ──
    start_map = {}
    node_durs: Dict[str, float] = {}

    for t in traces:
        stage = t.get("stage", "")
        action = t.get("action", "")
        if not action:
            continue

        if stage == "node_start":
            start_map[action] = t.get("timestamp", 0)

        elif stage == "node_end":
            ts_now = t.get("timestamp", 0)
            dur = t.get("duration_ms", 0.0) or 0.0
            if dur > 0:
                # 直接使用记录耗时
                node_durs[action] = node_durs.get(action, 0) + dur
            else:
                # 从 timestamp 推算
                started = start_map.pop(action, None)
                if started is not None:
                    dur = max(0, (ts_now - started) * 1000)
                    node_durs[action] = node_durs.get(action, 0) + dur

    # ── 最慢节点 ──
    slowest_node = ""
    if node_durs:
        slowest_node = max(node_durs, key=node_durs.get)

    return {
        "duration_s": _dur,
        "retry_count": retry_count,
        "retry_exhausted_count": retry_exhausted_count,
        "degraded_count": degraded_count,
        "resume_count": resume_count,
        "node_durations": {k: round(v / 1000, 2) for k, v in node_durs.items()},
        "slowest_node": slowest_node,
    }
