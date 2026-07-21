"""
BenchmarkRunner — ResearchForge 可重复评测框架

支持 Fast / Standard / Deep 三种模式的自动化评测，
记录统一 stats 供后续比较。
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from researchforge.research_service import ResearchService
from researchforge.orchestration import ResearchMode, ResearchGraph
from researchforge.orchestration.checkpoint_store import CheckpointStore
from researchforge.trace import TraceCollector

_CASES_DIR = Path(__file__).resolve().parent / "cases"


def load_cases(case_ids: Optional[List[str]] = None) -> List[dict]:
    """
    加载 Benchmark Case 文件。

    Args:
        case_ids: 指定 case id 列表，None 时加载全部

    Returns:
        case dict 列表
    """
    if not _CASES_DIR.exists():
        return []

    all_cases = []
    for fpath in sorted(_CASES_DIR.glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            case = json.load(f)
        all_cases.append(case)

    if case_ids:
        id_set = set(case_ids)
        all_cases = [c for c in all_cases if c.get("id") in id_set]

    return all_cases


def run_single_mode(
    case: dict,
    mode: str,
    llm: Any,
    checkpoint_store: Optional[CheckpointStore] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对单个 case + mode 执行一次研究并记录结果。

    Args:
        case: case dict
        mode: "fast" | "standard" | "deep"
        llm: LLMProvider 实例（mock 或真实）
        checkpoint_store: 可选检查点存储
        task_id: 可选任务 ID

    Returns:
        该次运行的 benchmark 结果 dict
    """
    mode_enum = ResearchMode(mode)
    tracer = TraceCollector(run_id=task_id or f"bm_{case['id']}_{mode}")

    start = time.time()
    try:
        svc = ResearchService(llm=llm, checkpoint_store=checkpoint_store)
        result = svc.run(
            topic=case["topic"],
            mode=mode_enum,
            tracer=tracer,
            task_id=tracer.run_id,
        )
        success = True
        error = None
    except Exception as e:
        result = {}
        success = False
        error = str(e)

    elapsed = round(time.time() - start, 2)

    # 从 result 中提取 stats
    stats = result.get("stats", {}) if result else {}
    trace_events = tracer.get_all() if tracer else []

    return {
        "mode": mode,
        "success": success,
        "error": error,
        "duration_s": elapsed,
        "stats": stats,
        "trace_count": len(trace_events),
    }


def run_benchmark(
    case: dict,
    llm: Any,
    modes: Optional[List[str]] = None,
    checkpoint_store: Optional[CheckpointStore] = None,
) -> Dict[str, Any]:
    """
    对单个 case 执行指定模式的 Benchmark。

    Args:
        case: case dict（含 id, topic, modes 等）
        llm: LLMProvider 实例
        modes: 要运行的模式列表，默认读取 case["modes"]
        checkpoint_store: 可选

    Returns:
        完整的 benchmark 结果 dict
    """
    modes = modes or case.get("modes", ["fast"])
    case_id = case.get("id", "unknown")
    topic = case.get("topic", "")

    results: Dict[str, Any] = {}
    for mode in modes:
        tid = f"bm_{case_id}_{mode}"
        mode_result = run_single_mode(
            case=case,
            mode=mode,
            llm=llm,
            checkpoint_store=checkpoint_store,
            task_id=tid,
        )
        results[mode] = mode_result

    return {
        "case": case_id,
        "topic": topic,
        "description": case.get("description", ""),
        "results": results,
    }
