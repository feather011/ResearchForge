"""
EvaluationMetrics — 统一质量统计结构

Fast、Standard、Deep 三种模式返回相同结构的 stats，
便于后续 Benchmark 和跨模式比较。
"""

import re
from typing import Any, Dict, List, Optional


def _build_citation_pattern(valid_source_ids: set) -> str:
    """
    构建引用标记的正则匹配模式。

    策略：
    1. 始终匹配标准 [来源N] 格式（Writer 生成格式）。
    2. 额外匹配每个有效 Source ID 的中括号包裹形式，用 re.escape 防特殊字符。
    3. 过滤掉与标准格式重复的模式。
    """
    patterns = [r'\[来源\d+\]']

    for sid in valid_source_ids:
        # 如果该 ID 已被标准格式覆盖则跳过
        if re.fullmatch(r'来源\d+', sid):
            continue
        patterns.append(r'\[' + re.escape(sid) + r'\]')

    # 多个模式用 | 组合
    return '|'.join(patterns) if patterns else r'(?!x)x'  # 永不匹配


def calculate_citation_metrics(report: str, sources: List) -> Dict[str, Any]:
    """
    计算报告引用质量指标。

    识别项目中实际使用的引用标记 [来源X]（Writer 输出格式），
    以及匹配任意有效 Source ID 的准确中括号引用。

    引用契约：
      - Writer 写入: [来源1], [来源2], ...（来源 ID = s.id，Writer 用 f"[{s.id}]" 写入）
      - SearchNode 生成: s.id = f"来源{sid}"
      - Audit 校验: re.findall(r'\\[来源\\d+\\]', report) 并验证 s.id in valid_ids

    Args:
        report: 研究报告文本
        sources: state.sources 列表（每个元素有 .id 属性）

    Returns:
        引用质量统计 dict
    """
    report = report or ""
    valid_source_ids = {s.id for s in (sources or [])}
    total_sources = len(valid_source_ids)

    # 构建匹配模式并提取引用标记
    pattern = _build_citation_pattern(valid_source_ids)
    all_marks = re.findall(pattern, report) if pattern else []
    total_marks = len(all_marks)

    if total_marks == 0:
        return {
            "total_marks": 0,
            "valid_marks": 0,
            "invalid_marks": 0,
            "unique_sources_cited": 0,
            "total_sources": total_sources,
            "valid_rate": 0.0,
            "source_utilization_rate": 0.0,
        }

    unique_cited = set()
    valid_count = 0
    invalid_count = 0

    for mark in all_marks:
        sid = mark.strip("[]")
        if sid in valid_source_ids:
            valid_count += 1
            unique_cited.add(sid)
        else:
            invalid_count += 1

    valid_rate = round(valid_count / total_marks, 4)
    util_rate = round(len(unique_cited) / total_sources, 4) if total_sources > 0 else 0.0

    return {
        "total_marks": total_marks,
        "valid_marks": valid_count,
        "invalid_marks": invalid_count,
        "unique_sources_cited": len(unique_cited),
        "total_sources": total_sources,
        "valid_rate": valid_rate,
        "source_utilization_rate": util_rate,
    }


def build_evaluation_stats(
    state: Any,
    traces: List[dict],
    duration_s: float,
    mode: str,
    audit_passed: bool = True,
    audit_rewritten: int = 0,
    audit_issues: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    构建统一的质量统计结构。

    纯函数，不从 state 重新执行任何评估逻辑。
    所有业务逻辑（Claim Verification / Audit / Coverage）的结果
    已经写入 state，本函数只做聚合统计。

    Args:
        state: ResearchState 实例
        traces: 全部 trace 事件列表
        duration_s: 执行耗时（秒）
        mode: "fast" | "standard" | "deep"
        audit_passed: 审计是否通过
        audit_rewritten: 重写次数
        audit_issues: 审计问题列表

    Returns:
        统一结构的 stats dict
    """

    # ── 基本计数 ──
    total_claims = len(state.claims) if hasattr(state, "claims") else 0
    claims = state.claims if hasattr(state, "claims") else []

    # ── Claim 分布 ──
    supported = sum(1 for c in claims if c.confidence >= 0.9) if claims else 0
    partial = sum(1 for c in claims if c.confidence == 0.5) if claims else 0
    zero_conf = sum(1 for c in claims if c.confidence == 0.0) if claims else 0

    # unverified 从 metadata 获取
    meta = getattr(state, "metadata", None) or {}
    unverified = meta.get("claim_verify_unresolved_count", 0)
    # unsupported = 0.0 置信度中减去 unverified 部分
    unsupported = max(0, zero_conf - unverified)

    supported_rate = round(supported / total_claims, 4) if total_claims > 0 else 0.0

    # ── Deep 模式特有 ──
    workers = 0
    conflicts = 0
    if mode == "deep":
        deep_workers = getattr(state, "deep_workers", None) or []
        workers = len(deep_workers)
        conflicts = getattr(state, "conflicts", None) or []
        conflicts = len(conflicts)

    # ── 审计降级 ──
    degraded = bool(meta.get("audit_degraded", False))

    # ── 证据覆盖率（从 metadata 读取，不重新执行） ──
    cov_evaluated = bool(meta.get("coverage_evaluated", False))
    cov_total = meta.get("coverage_total_questions", 0)
    cov_gap_count = meta.get("coverage_gap_count", 0)

    if cov_evaluated:
        if cov_total > 0:
            cov_covered = cov_total - cov_gap_count
            cov_rate = round(cov_covered / cov_total, 4)
        else:
            cov_covered = 0
            cov_gap_count = 0
            cov_rate = 0.0
    else:
        cov_covered = 0
        cov_rate = None

    # ── 执行指标（从 trace 聚合） ──
    from .execution_metrics import build_execution_metrics
    execution = build_execution_metrics(traces, duration_s=duration_s)

    # ── 引用质量 ──
    report_text = state.report if hasattr(state, "report") else ""
    sources_list = state.sources if hasattr(state, "sources") else []
    citation = calculate_citation_metrics(report_text, sources_list)

    # 先构建 stats dict，再计算 quality（依赖 stats 中 claims/citation/coverage/audit）
    _stats = {
        "sources": len(state.sources) if hasattr(state, "sources") else 0,
        "documents": len(state.documents) if hasattr(state, "documents") else 0,
        "evidences": len(state.evidences) if hasattr(state, "evidences") else 0,
        "report_length": len(state.report) if hasattr(state, "report") else 0,
        "claims": {
            "total": total_claims,
            "supported": supported,
            "partially_supported": partial,
            "unsupported": unsupported,
            "unverified": unverified,
            "supported_rate": supported_rate,
        },
        "coverage": {
            "evaluated": cov_evaluated,
            "total_questions": cov_total,
            "covered_questions": cov_covered,
            "gap_count": cov_gap_count,
            "coverage_rate": cov_rate,
        },
        "citation": citation,
        "audit": {
            "passed": audit_passed,
            "rewritten": audit_rewritten,
            "issues_count": len(audit_issues or []),
            "degraded": degraded,
        },
        "execution": execution,
        "workers": workers,
        "conflicts": conflicts,
    }

    # ── 质量评分（依赖已构建的 stats） ──
    from .quality_score import build_quality_score
    _stats["quality"] = build_quality_score(_stats)

    return _stats
