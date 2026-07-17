"""
metrics — 评估指标定义

所有指标都是纯函数，从 ResearchGraph 的执行结果 + tracer 事件中提取。
"""

import time
from typing import Dict, Any, List, Optional


def extract_metrics(
    result: Dict[str, Any],
    tracer_events: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
) -> Dict[str, Any]:
    """
    从单次研究执行中提取所有指标

    参数:
      result: ResearchGraph.execute() 的返回值
      tracer_events: TraceCollector.get_all() 的返回值
      start_time: 开始时间 (time.time)
      end_time: 结束时间 (time.time)

    返回:
      {
        "topic": str,
        "mode": str,
        "performance": {...},
        "research_quality": {...},
      }
    """

    # ── Performance ──
    total_time = round(end_time - start_time, 2)

    # 从 tracer events 中统计 LLM 调用次数和工具调用次数
    llm_calls = 0
    tool_calls = 0
    if tracer_events:
        for ev in tracer_events:
            if ev.get("stage") == "action" and ev.get("action") != "finish":
                tool_calls += 1
                if ev.get("tool_name"):
                    llm_calls += 1  # 每个 tool action 对应一次 LLM think

    # 从 result stats 补充
    llm_calls = max(llm_calls, 1)  # 至少有主流程的 LLM 调用

    # ── Research Quality ──
    stats = result.get("stats", {})

    # Claim 可信度统计
    claim_info = result.get("claim_verification", {})
    claim_total = claim_info.get("total", 0)
    claim_supported = claim_info.get("supported", 0)
    claim_unsupported = claim_info.get("unsupported", 0)

    # Audit 统计
    audit_info = result.get("audit", {})
    audit_passed = audit_info.get("passed", True)
    rewrite_count = audit_info.get("rewritten", 0)

    # 问题覆盖：通过 audit issues 反推
    total_questions = claim_total  # 近似
    question_coverage_rate = _calc_question_coverage(result, audit_info)

    return {
        "topic": result.get("topic", ""),
        "mode": result.get("mode", ""),
        "performance": {
            "total_time_seconds": total_time,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
        },
        "research_quality": {
            "question_coverage_rate": question_coverage_rate,
            "source_count": stats.get("sources", 0),
            "evidence_count": stats.get("evidences", 0),
            "claim_count": claim_total,
            "claim_supported_rate": round(claim_supported / max(claim_total, 1), 2),
            "audit_pass_rate": 1.0 if audit_passed else 0.0,
            "rewrite_count": rewrite_count,
            "report_length": stats.get("report_length", 0),
        },
    }


def _calc_question_coverage(
    result: Dict[str, Any],
    audit_info: Dict[str, Any],
) -> float:
    """
    估算问题覆盖率。
    使用 audit 结果和 claim 数量估算。
    """
    # 如果有 audit issues 涉及"未被覆盖"，降低覆盖率
    issues = audit_info.get("issues", [])
    uncovered_count = sum(1 for i in issues if "覆盖" in i or "覆盖" in i)

    claim_total = result.get("claim_verification", {}).get("total", 1)
    if uncovered_count > 0:
        return round(max(0, 1.0 - uncovered_count / max(claim_total, 1)), 2)

    # 没有 coverage issue → 100%
    return 1.0


def merge_mode_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并同一模式下的多次运行为平均值。
    """
    count = len(results)
    if count == 0:
        return {}

    perf_keys = ["total_time_seconds", "llm_calls", "tool_calls"]
    quality_keys = [
        "question_coverage_rate", "source_count", "evidence_count",
        "claim_count", "claim_supported_rate", "audit_pass_rate",
        "rewrite_count", "report_length",
    ]

    avg_perf = {}
    for k in perf_keys:
        vals = [r["performance"].get(k, 0) for r in results]
        avg_perf[k] = round(sum(vals) / count, 2)

    avg_quality = {}
    for k in quality_keys:
        vals = [r["research_quality"].get(k, 0) for r in results]
        avg_quality[k] = round(sum(vals) / count, 2)

    # 保留单次运行明细
    raw_runs = [
        {"topic": r["topic"], **r["performance"], **r["research_quality"]}
        for r in results
    ]

    return {
        "run_count": count,
        "average_performance": avg_perf,
        "average_quality": avg_quality,
        "runs": raw_runs,
    }
