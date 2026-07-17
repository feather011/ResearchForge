"""
Evaluation Runner — 对三种运行模式进行定量评估

用法:
    python evals/runner.py

流程:
    1. 加载 dataset.json（10 个研究主题）
    2. 对每个主题，依次用 Fast / Standard / Deep 模式执行
    3. 收集每条运行的 metrics
    4. 按模式合并，输出 evaluation_report.json

注意:
    - Deep 模式较慢，可配置 --skip-deep
    - 如果单主题失败，自动跳过（不影响其他主题）
    - 所有指标从现有 tracer / audit 数据计算
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

# 确保能从项目根目录 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from researchforge.core import BailianProvider
from researchforge.research_service import ResearchService
from researchforge.orchestration import ResearchMode
from researchforge.trace import TraceCollector

from evals.metrics import extract_metrics, merge_mode_results

# 配置
DATA_FILE = Path(__file__).parent / "dataset.json"
REPORT_FILE = Path(__file__) \
.parent / "evaluation_report.json"
TIMEOUT_PER_TOPIC = 600  # 每个主题最大等待秒数


def load_dataset() -> list:
    """加载评估数据集"""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single(
    service: ResearchService,
    topic: str,
    idx: int,
    mode: ResearchMode,
) -> dict:
    """
    执行一次研究，返回完整指标

    使用 tracer 记录全部事件，供 metrics 提取。
    """
    tracer = TraceCollector(run_id=f"eval_{idx}_{mode.value}")

    start = time.time()
    result = service.run(
        topic,
        mode=mode,
        progress_callback=None,  # 评估模式不推 SSE
        tracer=tracer,
    )
    end = time.time()

    # 提取指标
    metrics = extract_metrics(result, tracer.get_all(), start, end)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="ResearchForge Evaluation Runner")
    parser.add_argument("--skip-deep", action="store_true", help="跳过 Deep 模式（耗时较长）")
    parser.add_argument("--limit", type=int, default=10, help="运行前 N 个主题（默认 10）")
    args = parser.parse_args()

    dataset = load_dataset()[:args.limit]
    modes = [ResearchMode.FAST, ResearchMode.STANDARD]
    if not args.skip_deep:
        modes.append(ResearchMode.DEEP)

    print("=" * 60)
    print(f"ResearchForge Evaluation Runner")
    from researchforge.service.config import settings as _cfg
    from researchforge.core import OllamaProvider

    print(f"数据集: {len(dataset)} 个主题 × {len(modes)} 种模式")
    print(f"模型: {_cfg.MODEL} ({_cfg.LLM_PROVIDER})")
    print("=" * 60)

    # 初始化 LLM
    try:
        llm = OllamaProvider(
            model=_cfg.MODEL,
            base_url=_cfg.OLLAMA_BASE_URL,
            timeout=_cfg.LLM_TIMEOUT,
        )
    except Exception as e:
        print(f"[错误] LLM 初始化失败: {e}")
        sys.exit(1)

    service = ResearchService(llm=llm)

    # 按模式分组收集指标
    mode_results: dict = {m.value: [] for m in modes}

    for item in dataset:
        topic = item["topic"]
        idx = item["id"]

        print(f"\n[{idx}/{len(dataset)}] {topic}")

        for mode in modes:
            label = mode.value.upper()
            print(f"  {label}...", end=" ", flush=True)

            try:
                metrics = run_single(service, topic, idx, mode)
                mode_results[mode.value].append(metrics)
                print(f"OK ({metrics['performance']['total_time_seconds']:.0f}s)")
            except Exception as e:
                print(f"FAIL: {e}")
                continue

    # 合并结果
    report = {
        "config": {
            "dataset_size": len(dataset),
            "modes": [m.value for m in modes],
            "model": _cfg.MODEL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "modes": {},
        "summary": {},
    }

    best_mode = None
    best_score = -1

    for mode_name, results in mode_results.items():
        merged = merge_mode_results(results)
        report["modes"][mode_name] = merged

        # 综合评分（加权）
        avg_q = merged.get("average_quality", {})
        score = (
            avg_q.get("question_coverage_rate", 0) * 30 +
            avg_q.get("claim_supported_rate", 0) * 25 +
            avg_q.get("audit_pass_rate", 0) * 25 +
            min(avg_q.get("source_count", 0) / 5, 1.0) * 10 +
            min(avg_q.get("report_length", 0) / 1500, 1.0) * 10
        )
        merged["composite_score"] = round(score, 1)

        if score > best_score:
            best_score = score
            best_mode = mode_name

    report["summary"] = {
        "best_mode": best_mode,
        "best_score": best_score,
        "mode_comparison": {
            m: {
                "avg_time": report["modes"][m]["average_performance"]["total_time_seconds"],
                "avg_quality_score": report["modes"][m]["composite_score"],
            }
            for m in report["modes"]
        },
    }

    # 输出报告
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    for mode_name in [m.value for m in modes]:
        m = report["modes"].get(mode_name, {})
        ap = m.get("average_performance", {})
        aq = m.get("average_quality", {})
        print(f"\n{mode_name.upper()}:")
        print(f"  耗时: {ap.get('total_time_seconds', 'N/A')}s")
        print(f"  来源数: {aq.get('source_count', 'N/A')}")
        print(f"  证据数: {aq.get('evidence_count', 'N/A')}")
        print(f"  结论数: {aq.get('claim_count', 'N/A')}")
        print(f"  结论可信率: {aq.get('claim_supported_rate', 'N/A')}")
        print(f"  问题覆盖率: {aq.get('question_coverage_rate', 'N/A')}")
        print(f"  Audit通过率: {aq.get('audit_pass_rate', 'N/A')}")
        print(f"  综合评分: {m.get('composite_score', 'N/A')}")

    print(f"\n最佳模式: {report['summary']['best_mode']}")
    print(f"报告已保存: {REPORT_FILE}")


if __name__ == "__main__":
    main()
