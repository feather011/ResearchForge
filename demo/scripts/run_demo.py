"""
ResearchForge Demo Runner

支持 Mock 和真实两种运行方式，覆盖 Fast / Standard / Deep 三种模式。

用法:
    # Mock 运行（无需 API Key）
    python demo/scripts/run_demo.py --case fast --mock
    python demo/scripts/run_demo.py --case standard --mock
    python demo/scripts/run_demo.py --case deep --mock
    python demo/scripts/run_demo.py --all --mock

    # 故障恢复演示
    python demo/scripts/run_demo.py --case fast --mock --inject-fault

    # 真实运行（需要 API Key）
    python demo/scripts/run_demo.py --case fast
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))


def load_demo_cases() -> list:
    cases_path = Path(__file__).resolve().parent.parent / "demo_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_case(case_id: str) -> dict:
    cases = load_demo_cases()
    for c in cases:
        if c["id"] == case_id:
            return c
    raise ValueError(f"Case not found: {case_id}. Available: {[c['id'] for c in cases]}")


def setup_output_dir(run_id: str) -> Path:
    output_dir = Path(__file__).resolve().parent.parent / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_output(output_dir: Path, filename: str, content: str):
    (output_dir / filename).write_text(content, encoding="utf-8")


def run_mock_demo(case: dict, inject_fault: bool = False) -> dict:
    """使用 Mock LLM 运行 Demo（无需 API Key）"""
    from unittest.mock import patch
    import researchforge.nodes.search_node as search_mod
    import researchforge.nodes.fetch_node as fetch_mod
    import researchforge.nodes.extract_node as extract_mod
    import researchforge.nodes.synthesis_node as synthesis_mod
    import researchforge.nodes.write_node as write_mod
    import researchforge.nodes.claim_verification_node as cv_mod
    import researchforge.nodes.coverage_node as cov_mod
    import researchforge.nodes.audit_node as audit_mod

    from demo.mock_provider import (
        create_mock_search, create_mock_fetch, create_mock_extract,
        create_mock_claim_verify,
    )

    search_mod.run_search_node = create_mock_search()
    fetch_mod.run_fetch_node = create_mock_fetch()
    extract_mod.run_extract_node = create_mock_extract()
    cv_mod.run_claim_verification_node = create_mock_claim_verify()
    synthesis_mod.run_synthesis_node = (
        lambda state, llm: [
            __import__("researchforge.orchestration.research_state", fromlist=["Claim"]).Claim(
                text="该技术通过优化内存访问和计算图来提升性能", evidence_ids=["ev_0", "ev_1"]
            ),
            __import__("researchforge.orchestration.research_state", fromlist=["Claim"]).Claim(
                text="主流方案在保持精度的同时显著提升了推理速度", evidence_ids=["ev_0"]
            ),
        ]
    )
    write_mod.run_write_node = (
        lambda state, llm=None, extra_instructions="", mode="fast":
        "## 技术概述\n\n该技术是重要的研究方向。[来源1] 提出了核心实现方案。\n\n"
        "## 核心发现\n\n实验数据[来源1]表明该方法表现优异。[来源3]也验证了其可行性。\n\n## 结论\n\n具有重要应用价值。"
    )
    cov_mod.run_coverage_node = lambda state: (True, [])
    audit_mod.run_audit_node = lambda state, llm: type("AuditResult", (), {
        "passed": True, "issues": [], "suggestions": ""
    })()

    if inject_fault:
        from demo.scripts.fault_injector import get_fault_injector
        injector = get_fault_injector()
        injector.reset()
        injector.activate()
        injector.add_rule("searching", fail_count=1)
        injector.add_rule("deep_worker", fail_count=1, worker_id="W2")
        print("  ⚡ Fault injection enabled: searching × 1, WorkerW2 × 1")
        # Replace search node with fault-injecting wrapper
        _original_search = search_mod.run_search_node

        def _faulty_search(q, **kw):
            from demo.scripts.fault_injector import with_fault_tolerance
            with_fault_tolerance("searching")
            return _original_search(q, **kw)

        search_mod.run_search_node = _faulty_search

    from researchforge.research_service import ResearchService
    from researchforge.orchestration import ResearchMode
    from researchforge.trace import TraceCollector

    from demo.mock_provider import create_mock_llm
    llm = create_mock_llm()

    mode = case["modes"][0]
    run_id = f"{case['id']}_{mode}_{int(time.time())}"

    tracer = TraceCollector(run_id=run_id)
    svc = ResearchService(llm=llm)

    print(f"  Mode: {mode}")
    print(f"  Run ID: {run_id}")

    start = time.time()
    try:
        result = svc.run(
            topic=case["topic"],
            mode=ResearchMode(mode),
            tracer=tracer,
            task_id=run_id,
        )
        status = "completed"
        error = None
        duration = round(time.time() - start, 2)
        print(f"  Status: ✅ {status} ({duration}s)")
    except Exception as e:
        result = {}
        status = "failed"
        error = str(e)
        duration = round(time.time() - start, 2)
        print(f"  Status: ❌ {status} — {error}")

    # 收集输出
    output = {
        "case": case["id"],
        "topic": case["topic"],
        "mode": mode,
        "status": status,
        "duration_s": duration,
        "error": error,
        "result": {
            "report": result.get("report", ""),
            "stats": result.get("stats", {}),
        },
        "traces": tracer.get_all() if tracer else [],
    }

    return output


def run_real_demo(case: dict) -> dict:
    """使用真实 LLM 运行 Demo（需要 API Key）"""
    from researchforge.research_service import ResearchService
    from researchforge.orchestration import ResearchMode
    from researchforge.trace import TraceCollector
    from researchforge.service.config import settings

    # 检查配置
    if not settings.DASHSCOPE_API_KEY or settings.DASHSCOPE_API_KEY == "sk-your-api-key-here":
        print("  ❌ DASHSCOPE_API_KEY 未配置。请先编辑 .env 文件。")
        print("  💡 使用 --mock 参数可无 API Key 运行。")
        sys.exit(1)

    from researchforge.core import BailianProvider, OllamaProvider

    if settings.LLM_PROVIDER == "ollama":
        llm = OllamaProvider(model=settings.MODEL, base_url=settings.OLLAMA_BASE_URL, timeout=settings.LLM_TIMEOUT)
    else:
        llm = BailianProvider(model=settings.MODEL, timeout=settings.LLM_TIMEOUT)

    mode = case["modes"][0]
    if case["modes"][0] == "deep" and settings.LLM_PROVIDER == "ollama":
        print("  ⚠️ Deep 模式在 Ollama 上可能需要较长时间，建议使用 Bailian。")

    run_id = f"{case['id']}_{mode}_{int(time.time())}"
    tracer = TraceCollector(run_id=run_id)
    svc = ResearchService(llm=llm)

    print(f"  Mode: {mode}")
    print(f"  Run ID: {run_id}")

    start = time.time()
    try:
        result = svc.run(
            topic=case["topic"],
            mode=ResearchMode(mode),
            tracer=tracer,
            task_id=run_id,
        )
        status = "completed"
        error = None
        duration = round(time.time() - start, 2)
        print(f"  Status: ✅ {status} ({duration}s)")
    except Exception as e:
        result = {}
        status = "failed"
        error = str(e)
        duration = round(time.time() - start, 2)
        print(f"  Status: ❌ {status} — {error}")
        traceback.print_exc()

    output = {
        "case": case["id"],
        "topic": case["topic"],
        "mode": mode,
        "status": status,
        "duration_s": duration,
        "error": error,
        "result": {
            "report": result.get("report", ""),
            "stats": result.get("stats", {}),
        },
        "traces": tracer.get_all() if tracer else [],
    }

    return output


def save_output(output: dict):
    run_id = f"{output['case']}_{output['mode']}_{int(time.time())}" if 'run_id' not in output else ""
    output_dir = setup_output_dir(run_id or f"{output['case']}_{output['mode']}")

    stats = output.get("result", {}).get("stats", {})

    # result.json
    write_output(output_dir, "result.json", json.dumps(output, ensure_ascii=False, indent=2))

    # report.md
    report = output.get("result", {}).get("report", "")
    if report:
        write_output(output_dir, "report.md", report)

    # traces.json
    traces = output.get("traces", [])
    if traces:
        write_output(output_dir, "traces.json",
                     json.dumps(traces, ensure_ascii=False, indent=2))

    # summary.md
    quality = stats.get("quality", {})
    claims = stats.get("claims", {})
    execution = stats.get("execution", {})
    citation = stats.get("citation", {})
    coverage = stats.get("coverage", {})
    audit = stats.get("audit", {})

    summary = f"""# Demo Run Summary

- **Case**: {output['case']}
- **Topic**: {output['topic']}
- **Mode**: {output['mode']}
- **Status**: {output['status']}
- **Duration**: {output['duration_s']}s
- **Error**: {output.get('error', 'None')}

## Stats

| Metric | Value |
|--------|-------|
| Sources | {stats.get('sources', 0)} |
| Evidences | {stats.get('evidences', 0)} |
| Report Length | {stats.get('report_length', 0)} |
| Claims Total | {claims.get('total', 0)} |
| Supported | {claims.get('supported', 0)} |
| Supported Rate | {claims.get('supported_rate', 0)} |
| Quality Score | {quality.get('quality_score', 'N/A')} |
| Quality Grade | {quality.get('grade', 'N/A')} |
| Citation Valid Rate | {citation.get('valid_rate', 0)} |
| Citation Utilization | {citation.get('source_utilization_rate', 0)} |
| Coverage Evaluated | {coverage.get('evaluated', False)} |
| Coverage Rate | {coverage.get('coverage_rate', 'N/A')} |
| Audit Passed | {audit.get('passed', 'N/A')} |
| Audit Degraded | {audit.get('degraded', False)} |
| Duration | {execution.get('duration_s', 0)}s |
| Retry Count | {execution.get('retry_count', 0)} |
| Degraded Count | {execution.get('degraded_count', 0)} |
| Slowest Node | {execution.get('slowest_node', 'N/A')} |
"""
    write_output(output_dir, "summary.md", summary)

    print(f"  📁 Output: {output_dir}/")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="ResearchForge Demo Runner")
    parser.add_argument("--case", type=str, default=None,
                        help="Case ID to run (fast_simple, standard_multi_source, deep_complex)")
    parser.add_argument("--all", action="store_true", help="Run all cases")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API key required)")
    parser.add_argument("--inject-fault", action="store_true",
                        help="Enable fault injection for retry/resume demo (mock only)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ResearchForge Demo")
    print("=" * 60)

    if args.all:
        cases = load_demo_cases()
    elif args.case:
        cases = [find_case(args.case)]
    else:
        # 默认运行 fast_simple
        cases = [find_case("fast_simple")]

    for case in cases:
        print(f"\n{'─' * 40}")
        print(f"  Case: {case['id']}")
        print(f"  Topic: {case['topic']}")
        print(f"{'─' * 40}")

        if args.mock:
            output = run_mock_demo(case, inject_fault=args.inject_fault)
        else:
            output = run_real_demo(case)

        save_output(output)

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
