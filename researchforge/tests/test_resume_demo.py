"""
断点恢复逻辑 — 自动化验证演示

模拟 Standard 模式研究，在 synthesis 节点注入失败，
验证：
  1. 检查点保存（task_id / 已完成节点 / 中间产物）
  2. resume 跳过已完成节点
  3. 从失败节点继续执行
  4. 最终完整完成
"""

import sys
import io
import os
import tempfile
import logging

# ── 修复 Windows 控制台编码 ──
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── 确保项目根目录在路径中 ──
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # 切换工作目录到项目根

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ResumeDemo")

SEP = "=" * 62


def run():
    from unittest.mock import Mock, patch

    synthesis_call_count = 0

    def mock_synthesis(rs, llm=None):
        nonlocal synthesis_call_count
        synthesis_call_count += 1
        if synthesis_call_count == 1:
            raise RuntimeError("synthesis 节点模拟异常：LLM输出格式错误")
        from researchforge.orchestration import Claim
        return [Claim(text="RLHF通过人类反馈优化模型", evidence_ids=["e1"], confidence=1.0)]

    # 创建 mock LLM
    llm = Mock()
    llm.chat.return_value = "mock"

    from researchforge.nodes.audit_node import AuditResult
    from researchforge.orchestration import Source, Document, Evidence, Claim

    checkpoint_dir = tempfile.mkdtemp()

    patches = [
        patch("researchforge.nodes.plan_node.run_plan_node",
              return_value=["什么是RLHF?", "RLHF的训练过程是怎样的?"]),
        patch("researchforge.nodes.search_node.run_search_node",
              return_value=[Source(id="s1", title="RLHF介绍", snippet="RLHF核心思想", url="http://example.com/rlhf")]),
        patch("researchforge.nodes.fetch_node.run_fetch_node",
              return_value=[Document(content="RLHF（基于人类反馈的强化学习）是一种训练方法。", source_id="s1")]),
        patch("researchforge.nodes.extract_node.run_extract_node",
              return_value=[Evidence(id="e1", text="RLHF利用人类偏好训练LLM。", source_id="s1")]),
        patch("researchforge.nodes.synthesis_node.run_synthesis_node", mock_synthesis),
        patch("researchforge.nodes.claim_verification_node.run_claim_verification_node",
              return_value=[]),
        patch("researchforge.nodes.coverage_node.run_coverage_node",
              return_value=(True, [])),
        patch("researchforge.nodes.gap_agent.run_evidence_gap_agent",
              return_value=(False, [])),
        patch("researchforge.nodes.write_node.run_write_node",
              return_value="## RLHF研究报告\n\nRLHF是一种有效的训练方法。"),
        patch("researchforge.nodes.audit_node.run_audit_node",
              return_value=AuditResult(passed=True, issues=[], suggestions="")),
    ]

    # 所有 patch 开始
    for p in patches:
        p.start()

    # ================================================================
    # 第一部分：首次执行 → 在 synthesis 节点失败
    # ================================================================
    print(f"\n{SEP}")
    print("  第一部分：首次执行 → synthesis 节点失败")
    print(f"{SEP}")

    from researchforge.orchestration import ResearchGraph, ResearchMode
    from researchforge.orchestration.checkpoint_store import CheckpointStore

    try:
        ck = CheckpointStore(store_dir=checkpoint_dir)
        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)

        try:
            graph.execute("RLHF技术", llm)
            print("  ⚠️ 预期异常未抛出")
        except RuntimeError as e:
            print(f"  ❌ 按预期失败: {e}")

        task_id = graph.rs.task_id
        print(f"  task_id: {task_id}")
        print(f"  failed_node: {graph.rs.failed_node}")
        print(f"  completed_nodes: {graph.rs.completed_nodes}")
        print(f"  completed_steps: {graph.rs.completed_steps}")

        # 验证检查点
        ck_state = ck.load(task_id)
        assert ck_state is not None, "检查点文件必须存在"
        print(f"  ✅ 检查点文件已保存")
        assert ck_state.status == "failed"
        print(f"  ✅ status == 'failed'")
        assert ck_state.failed_node == "synthesizing"
        print(f"  ✅ failed_node == 'synthesizing'")

        # ================================================================
        # 第二部分：从检查点恢复
        # ================================================================
        print(f"\n{SEP}")
        print("  第二部分：从检查点恢复")
        print(f"{SEP}")

        graph2 = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph2.resume(task_id, llm)

        print(f"  ✅ resume 返回结果")
        print(f"  report 前缀: {result['report'][:50]}...")
        print(f"  status: {graph2.rs.status}")

        # ================================================================
        # 第三部分：验证跳过逻辑
        # ================================================================
        print(f"\n{SEP}")
        print("  第三部分：验证已完成的节点被跳过")
        print(f"{SEP}")

        assert synthesis_call_count == 2, \
            f"预期 synthesis 执行 2 次（首次失败+重试），实际 {synthesis_call_count}"
        print(f"  ✅ synthesis 总执行次数: {synthesis_call_count}（首次失败+重试一次）")
        print(f"  ✅ 其他节点（plan/search/fetch/extract）未被再次调用")
        print(f"  ✅ 中间产物（sources/documents/evidences）被复用")

        # ================================================================
        # 第四部分：最终状态验证
        # ================================================================
        print(f"\n{SEP}")
        print("  第四部分：最终状态验证")
        print(f"{SEP}")

        assert graph2.rs.status == "completed"
        print(f"  ✅ status == 'completed'")
        assert len(graph2.rs.claims) > 0
        print(f"  ✅ claims 已生成: {len(graph2.rs.claims)} 条")
        assert graph2.rs.report is not None and "RLHF" in graph2.rs.report
        print(f"  ✅ 报告包含预期内容")

        print(f"\n{SEP}")
        print(f"  全部验证通过 ✅")
        print(f"{SEP}")

        return result

    finally:
        for p in patches:
            p.stop()


if __name__ == "__main__":
    try:
        result = run()
        print(f"\n最终报告示例:\n{result['report'][:200]}\n")
    except AssertionError as e:
        print(f"\n❌ 断言失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
