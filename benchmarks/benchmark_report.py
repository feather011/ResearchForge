"""
BenchmarkReport — Benchmark 结果报告生成

将 run_benchmark 的输出转换为对比报告。
"""

from typing import Any, Dict, List, Optional


def build_report(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 benchmark 结果转换为对比报告。

    当前只展示原始指标，不设计 quality_score。

    Args:
        benchmark_results: run_benchmark 的返回值

    Returns:
        对比报告 dict
    """
    case = benchmark_results.get("case", "unknown")
    topic = benchmark_results.get("topic", "")
    raw_results = benchmark_results.get("results", {})

    comparison: Dict[str, Any] = {}

    for mode, mode_result in raw_results.items():
        stats = mode_result.get("stats", {})
        execution = stats.get("execution", {})
        claims = stats.get("claims", {})
        citation = stats.get("citation", {})
        coverage = stats.get("coverage", {})
        audit = stats.get("audit", {})

        entry = {
            "success": mode_result.get("success", False),
            "error": mode_result.get("error"),
            "duration_s": mode_result.get("duration_s", 0),
            "trace_count": mode_result.get("trace_count", 0),
            "stats": {
                "sources": stats.get("sources", 0),
                "documents": stats.get("documents", 0),
                "evidences": stats.get("evidences", 0),
                "report_length": stats.get("report_length", 0),
                "claims": {
                    "total": claims.get("total", 0),
                    "supported": claims.get("supported", 0),
                    "supported_rate": claims.get("supported_rate", 0.0),
                },
                "citation": {
                    "total_marks": citation.get("total_marks", 0),
                    "valid_rate": citation.get("valid_rate", 0.0),
                    "source_utilization_rate": citation.get("source_utilization_rate", 0.0),
                },
                "coverage": {
                    "evaluated": coverage.get("evaluated", False),
                    "coverage_rate": coverage.get("coverage_rate"),
                },
                "audit": {
                    "passed": audit.get("passed", True),
                    "rewritten": audit.get("rewritten", 0),
                    "degraded": audit.get("degraded", False),
                },
                "quality": {
                    "score": stats.get("quality", {}).get("quality_score", 0.0),
                    "grade": stats.get("quality", {}).get("grade", "N/A"),
                },
            },
            "execution": {
                "duration_s": execution.get("duration_s", 0),
                "retry_count": execution.get("retry_count", 0),
                "degraded_count": execution.get("degraded_count", 0),
                "slowest_node": execution.get("slowest_node", ""),
            },
        }
        comparison[mode] = entry

    return {
        "case": case,
        "topic": topic,
        "comparison": comparison,
    }


def format_report_text(report: Dict[str, Any]) -> str:
    """
    将报告格式化为可读文本。

    Args:
        report: build_report 的返回值

    Returns:
        可打印的文本报告
    """
    lines = []
    lines.append(f"=== Benchmark: {report['case']} ===")
    lines.append(f"主题: {report['topic']}")
    lines.append("")

    for mode, data in report.get("comparison", {}).items():
        ok = "✅" if data["success"] else "❌"
        lines.append(f"  [{mode.upper()}] {ok}")
        if data.get("error"):
            lines.append(f"    错误: {data['error']}")
        lines.append(f"    耗时: {data['duration_s']}s")
        lines.append(f"    来源: {data['stats']['sources']}")
        lines.append(f"    证据: {data['stats']['evidences']}")
        lines.append(f"    报告: {data['stats']['report_length']} 字")
        lines.append(f"    Claims: {data['stats']['claims']['total']} ({data['stats']['claims']['supported']} 可信)")
        lines.append(f"    引用: {data['stats']['citation']['total_marks']} 标记, 有效率 {data['stats']['citation']['valid_rate']}")
        lines.append(f"    重试: {data['execution']['retry_count']}")
        lines.append(f"    降级: {data['execution']['degraded_count']}")
        lines.append(f"    最慢节点: {data['execution']['slowest_node']}")
        lines.append(f"    质量评分: {data['stats']['quality']['score']} ({data['stats']['quality']['grade']})")
        lines.append("")

    return "\n".join(lines)
