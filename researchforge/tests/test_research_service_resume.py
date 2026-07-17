"""
ResearchService 恢复功能测试
"""

import tempfile
from pathlib import Path

import pytest
from researchforge.orchestration import ResearchGraph, ResearchMode
from researchforge.orchestration.checkpoint_store import CheckpointStore
from researchforge.research_service import ResearchService


class ResearchServiceWithCk(ResearchService):
    """注入 checkpoint_store 的子类（避免改 ResearchService 构造函数签名）"""
    pass


class TestResearchServiceResume:
    """测试 ResearchService.resume()"""

    @pytest.fixture
    def ck(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield CheckpointStore(store_dir=Path(tmp))

    @pytest.fixture
    def svc(self, ck):
        return ResearchService(llm=None, checkpoint_store=ck)

    # ── 辅助：mock 所有节点 ──

    @staticmethod
    def _mock_all_nodes(monkeypatch):
        """Fast 模式 mock"""
        from researchforge.nodes.plan_node import run_plan_node
        from researchforge.nodes.search_node import run_search_node
        from researchforge.nodes.fetch_node import run_fetch_node
        from researchforge.nodes.extract_node import run_extract_node
        from researchforge.nodes.synthesis_node import run_synthesis_node
        from researchforge.nodes.write_node import run_write_node

        def mock_plan(t, llm=None): return ["q1"]
        def mock_search(q, **kw):
            from researchforge.orchestration import Source
            return [Source(id="s1", title="M", snippet="m", url="")]
        def mock_fetch(src, **kw):
            from researchforge.orchestration import Document
            return [Document(content="c", source_id="s1")]
        def mock_extract(docs, q):
            from researchforge.orchestration import Evidence
            return [Evidence(id="e1", text="mock evidence", source_id="s1")]
        def mock_synthesis(rs, llm=None):
            from researchforge.orchestration import Claim
            return [Claim(text="c1", evidence_ids=["e1"], confidence=1.0)]
        def mock_write(rs, llm=None, extra_instructions="", mode="standard"):
            return "恢复测试报告"

        monkeypatch.setattr("researchforge.nodes.plan_node.run_plan_node", mock_plan)
        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node", mock_search)
        monkeypatch.setattr("researchforge.nodes.fetch_node.run_fetch_node", mock_fetch)
        monkeypatch.setattr("researchforge.nodes.extract_node.run_extract_node", mock_extract)
        monkeypatch.setattr("researchforge.nodes.synthesis_node.run_synthesis_node", mock_synthesis)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write)

    # ── 测试 ──

    def test_resume_standard_success(self, monkeypatch, svc, ck):
        """Standard 模式完整执行后模拟中断，恢复成功"""
        self._mock_all_nodes(monkeypatch)
        from researchforge.nodes.audit_node import AuditResult

        def mock_audit(rs, llm):
            return AuditResult(passed=True, issues=[])
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit)

        # 第一次完整执行
        result = svc.run("测试", mode=ResearchMode.STANDARD)
        task_id = result["_task_id"] = None  # ResearchService.run 不返回 task_id

        # 从 checkpoint 找到最近的任务 ID
        ids = ck.list_ids()
        assert len(ids) > 0
        task_id = ids[-1]

        # 模拟中断（修改检查点状态为 failed）
        saved = ck.load(task_id)
        saved.status = "failed"
        saved.failed_node = ""
        ck.save(saved)

        # 恢复
        result2 = svc.resume(task_id)
        assert result2["report"] == "恢复测试报告"
        assert result2["mode"] == "standard"

    def test_resume_nonexistent_raises(self, svc):
        """不存在的检查点恢复时抛出 ValueError"""
        with pytest.raises(ValueError, match="不存在"):
            svc.resume("nonexistent_id")

    def test_resume_completed_raises(self, monkeypatch, svc, ck):
        """已完成的任务恢复时抛出 ValueError"""
        self._mock_all_nodes(monkeypatch)
        svc.run("测试", mode=ResearchMode.FAST)

        ids = ck.list_ids()
        assert len(ids) > 0

        with pytest.raises(ValueError, match="已完成"):
            svc.resume(ids[-1])

    def test_resume_deep_raises(self, monkeypatch, svc, ck):
        """Deep 模式恢复时抛出 NotImplementedError"""
        # 创建一个 Deep 模式的检查点（直接存一个 Deep 状态）
        from researchforge.orchestration import ResearchState
        state = ResearchState(mode=ResearchMode.DEEP, topic="deep测试", task_id="deep_test")
        ck.save(state)

        with pytest.raises(NotImplementedError, match="不支持"):
            svc.resume("deep_test")

    def test_resume_without_store_raises(self):
        """未配置 CheckpointStore 时恢复报错"""
        svc = ResearchService(llm=None)
        with pytest.raises(RuntimeError, match="未配置"):
            svc.resume("task_id")

    def test_new_task_still_works(self, monkeypatch, svc):
        """新增 resume 后，原有 run() 行为不变"""
        self._mock_all_nodes(monkeypatch)
        result = svc.run("新任务测试", mode=ResearchMode.FAST)
        assert result["report"] == "恢复测试报告"
        assert result["mode"] == "fast"
