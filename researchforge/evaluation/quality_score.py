"""
QualityScore — 基于已有 stats 的可复现质量评分

不新增 LLM 调用，不修改业务逻辑。
"""

from typing import Any, Dict, Optional


def _claim_score(stats: dict) -> int:
    """Claim 维度 30%：基于 supported_rate"""
    claims = stats.get("claims", {})
    rate = claims.get("supported_rate", 0.0) or 0.0
    if rate >= 0.9:
        return 100
    elif rate >= 0.7:
        return 80
    elif rate >= 0.5:
        return 60
    else:
        return 40


def _citation_score(stats: dict) -> int:
    """Citation 维度 25%：valid_rate 占 70% + source_utilization_rate 占 30%"""
    citation = stats.get("citation", {})
    valid_rate = citation.get("valid_rate", 0.0) or 0.0
    util_rate = citation.get("source_utilization_rate", 0.0) or 0.0
    return round(valid_rate * 70 + util_rate * 30)


def _coverage_score(stats: dict) -> int:
    """Coverage 维度 25%：已评估用 coverage_rate，未评估给 50"""
    coverage = stats.get("coverage", {})
    if coverage.get("evaluated"):
        rate = coverage.get("coverage_rate") or 0.0
        return round(rate * 100)
    else:
        return 50


def _audit_score(stats: dict) -> int:
    """Audit 维度 20%：passed→100, degraded→50, issues 每个扣 10"""
    audit = stats.get("audit", {})
    if audit.get("degraded"):
        base = 50
    elif audit.get("passed"):
        base = 100
    else:
        base = 50

    issues_count = audit.get("issues_count", 0)
    penalty = min(issues_count * 10, base)
    return max(0, base - penalty)


def build_quality_score(stats: dict) -> Dict[str, Any]:
    """
    基于已有 stats 计算可复现质量评分。

    Args:
        stats: 统一 stats dict（来自 build_evaluation_stats）

    Returns:
        质量评分 dict，包含 quality_score, breakdown, grade
    """
    if not stats:
        return {
            "quality_score": 0.0,
            "grade": "F",
            "breakdown": {
                "claim_score": 0,
                "citation_score": 0,
                "coverage_score": 0,
                "audit_score": 0,
            },
        }

    c_score = _claim_score(stats)
    ci_score = _citation_score(stats)
    co_score = _coverage_score(stats)
    a_score = _audit_score(stats)

    total = round(
        c_score * 0.30 +
        ci_score * 0.25 +
        co_score * 0.25 +
        a_score * 0.20,
        1,
    )

    if total >= 90:
        grade = "A"
    elif total >= 75:
        grade = "B"
    elif total >= 60:
        grade = "C"
    elif total >= 40:
        grade = "D"
    else:
        grade = "F"

    return {
        "quality_score": total,
        "grade": grade,
        "breakdown": {
            "claim_score": c_score,
            "citation_score": ci_score,
            "coverage_score": co_score,
            "audit_score": a_score,
        },
    }
