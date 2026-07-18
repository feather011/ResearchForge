"""
Deep 模式 Checkpoint 单元测试
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from researchforge.orchestration import ResearchMode, ResearchState, Source, Document, Evidence
from researchforge.orchestration.checkpoint_store import CheckpointStore
from researchforge.orchestration.research_state import DeepWorkerState


@pytest.fixture
def ck():
    """CheckpointStore 临时目录（供所有测试类共用）"""
    with tempfile.TemporaryDirectory() as tmp:
        yield CheckpointStore(store_dir=Path(tmp))


class TestDeepCheckpoint:
    """Deep 模式 Checkpoint 测试"""

    def _mock_llm(self):
        """返回一个可用的 Mock LLM（generate 返回简单的子任务）"""
        m = Mock()
        m.generate.return_value = "1. 子任务A\n2. 子任务B"
        return m

    def test_deep_plan_checkpoint_exists(self, monkeypatch, ck):
        """Deep Planning 后存在 Checkpoint"""
        from researchforge.research_service import ResearchService

        llm = self._mock_llm()

        # Mock Worker.run 快速返回
        def mock_worker_run(worker_self):
            return {"worker_id": worker_self.worker_id, "task": worker_self.task,
                    "sources": [], "documents": [], "evidences": [], "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_worker_run)

        svc = ResearchService(llm=llm, checkpoint_store=ck)
        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.gap_agent.run_evidence_gap_agent", return_value=(False, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.run("测试", mode=ResearchMode.DEEP)

        # Planning 后的 checkpoint 应存在
        # task_id 由 svc.run 内部生成 → 从返回结果无法直接获取
        # 但 CheckpointStore 应该至少保存了 1 个检查点
        assert ck.count() >= 1, "Deep 执行后应至少存在 1 个检查点"

    def test_deep_worker_state_saved(self, monkeypatch, ck):
        """单个 Worker 完成后结果已保存"""
        from researchforge.research_service import ResearchService

        llm = self._mock_llm()

        def mock_worker_run(worker_self):
            from researchforge.orchestration import Source, Document, Evidence
            return {"worker_id": worker_self.worker_id, "task": worker_self.task,
                    "sources": [Source(id="s1", title="S1", snippet="s1", url="")],
                    "documents": [Document(content="doc1", source_id="s1")],
                    "evidences": [Evidence(id="e1", source_id="s1", text="ev1")],
                    "claims": ["结论1"]}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_worker_run)

        svc = ResearchService(llm=llm, checkpoint_store=ck)
        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.gap_agent.run_evidence_gap_agent", return_value=(False, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.run("测试", mode=ResearchMode.DEEP)

        # 列出所有检查点，load 最后一个查看 workers
        ids = ck.list_ids()
        assert len(ids) > 0
        state = ck.load(ids[-1])
        assert state is not None
        # deep_workers 应有记录
        assert len(state.deep_workers) > 0, "deep_workers 应有记录"
        # 至少有一个 worker 状态为 completed
        completed = [w for w in state.deep_workers if w.status == "completed"]
        assert len(completed) >= 1, "至少有一个 worker 状态为 completed"
        # completed worker 应有 sources
        if completed:
            assert len(completed[0].sources) > 0, "completed worker 应有 sources"

    def test_deep_workers_dont_overwrite(self, monkeypatch, ck):
        """多 Worker 并发保存不会互相覆盖"""
        from researchforge.research_service import ResearchService
        import time

        llm = self._mock_llm()

        call_order = []

        def mock_worker_run(worker_self):
            time.sleep(0.1)  # 保证并发
            call_order.append(worker_self.worker_id)
            return {"worker_id": worker_self.worker_id, "task": worker_self.task,
                    "sources": [Source(id=f"s_{worker_self.worker_id}", title="", snippet="", url="")],
                    "documents": [], "evidences": [], "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_worker_run)

        svc = ResearchService(llm=llm, checkpoint_store=ck)
        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.deep_research.LeadResearcher.make_plan",
                   return_value=["任务A", "任务B"]), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.run("测试", mode=ResearchMode.DEEP)

        # 最终检查点应有 2 个 worker 都 completed
        ids = ck.list_ids()
        state = ck.load(ids[-1])
        completed_ids = [w.worker_id for w in state.deep_workers if w.status == "completed"]
        assert len(completed_ids) == 2, f"2 个 worker 都应 completed, 实际: {completed_ids}"
        # 每个 worker 有自己的 source
        for w in state.deep_workers:
            if w.status == "completed":
                expected_id = f"s_{w.worker_id}"
                assert any(s.id == expected_id for s in w.sources), \
                    f"Worker {w.worker_id} 应有自己的 source {expected_id}"

    def test_deep_partial_failure_preserves_success(self, monkeypatch, ck):
        """部分 Worker 失败时成功结果仍保留"""
        from researchforge.research_service import ResearchService

        llm = self._mock_llm()
        fail_count = [0]

        def mock_worker_run(worker_self):
            fail_count[0] += 1
            if fail_count[0] == 1:
                raise RuntimeError("Worker1 模拟失败")
            return {"worker_id": worker_self.worker_id, "task": worker_self.task,
                    "sources": [Source(id=f"s_{worker_self.worker_id}", title="", snippet="", url="")],
                    "documents": [], "evidences": [], "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_worker_run)

        svc = ResearchService(llm=llm, checkpoint_store=ck)
        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.deep_research.LeadResearcher.make_plan",
                   return_value=["任务A", "任务B"]), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.run("测试", mode=ResearchMode.DEEP)

        ids = ck.list_ids()
        state = ck.load(ids[-1])
        # 确保有一个 failed 和一个 completed
        statuses = [w.status for w in state.deep_workers]
        assert "failed" in statuses, "应有 failed worker"
        assert "completed" in statuses, "应有 completed worker"
        # successful worker 的 source 必须保留
        for w in state.deep_workers:
            if w.status == "completed":
                assert len(w.sources) > 0, f"completed worker {w.worker_id} 应有 sources"

    def test_deep_all_workers_completed_flag(self, monkeypatch, ck):
        """所有 Worker 完成后步骤状态正确"""
        from researchforge.research_service import ResearchService

        llm = self._mock_llm()

        def mock_worker_run(worker_self):
            return {"worker_id": worker_self.worker_id, "task": worker_self.task,
                    "sources": [], "documents": [], "evidences": [], "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_worker_run)

        svc = ResearchService(llm=llm, checkpoint_store=ck)
        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.run("测试", mode=ResearchMode.DEEP)

        ids = ck.list_ids()
        state = ck.load(ids[-1])
        assert state.deep_workers_completed is True, "deep_workers_completed 应为 True"

    def test_deep_original_mode_still_works(self, monkeypatch, ck):
        """原 Deep 模式仍能正常运行（完整执行链）"""
        from researchforge.research_service import ResearchService

        llm = self._mock_llm()

        def mock_worker_run(worker_self):
            from researchforge.orchestration import Source, Document, Evidence
            return {"worker_id": worker_self.worker_id, "task": worker_self.task,
                    "sources": [Source(id="s1", title="S1", snippet="s1", url="")],
                    "documents": [Document(content="doc1", source_id="s1")],
                    "evidences": [Evidence(id="e1", source_id="s1", text="ev1")],
                    "claims": ["结论1"]}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_worker_run)

        svc = ResearchService(llm=llm, checkpoint_store=ck)
        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.run("测试", mode=ResearchMode.DEEP)

        assert result["mode"] == "deep"
        assert "report" in result
        assert result["stats"]["workers"] >= 1


class TestDeepResume:
    """Deep 模式 Resume 测试"""

    # ── 辅助：准备已完成的 Checkpoint ──

    def _prepare_deep_state(self, ck, worker_statuses=None):
        """创建一个带有 deep_workers 的 ResearchState"""
        from researchforge.orchestration.research_state import DeepWorkerState, ResearchState
        from researchforge.orchestration import Source

        import uuid
        rs = ResearchState(mode=ResearchMode.DEEP, topic="恢复测试", task_id=uuid.uuid4().hex[:8])
        rs.status = "failed"
        rs.questions = ["任务A", "任务B"]
        rs.deep_workers = [
            DeepWorkerState(worker_id="W1", task="任务A",
                            status=worker_statuses[0] if worker_statuses else "completed",
                            sources=[Source(id="s_w1", title="W1", snippet="w1", url="")],
                            documents=[], evidences=[], error=""),
            DeepWorkerState(worker_id="W2", task="任务B",
                            status=worker_statuses[1] if worker_statuses else "failed",
                            sources=[], documents=[], evidences=[], error="W2 failed"),
        ]
        rs.deep_workers_completed = False
        ck.save(rs)
        return rs.task_id

    def _mock_llm(self):
        m = Mock()
        m.generate.return_value = "1. 子任务A\n2. 子任务B"
        return m

    # ── 测试 ──

    def test_resume_skips_completed_workers(self, monkeypatch, ck):
        """已完成 Worker 在 Resume 后不重复执行"""
        from researchforge.research_service import ResearchService
        from researchforge.nodes.deep_research import ResearchWorker

        task_id = self._prepare_deep_state(ck, ["completed", "completed"])
        call_count = {"run": 0}

        original_run = ResearchWorker.run
        def mock_run(self_worker):
            call_count["run"] += 1
            return {"worker_id": self_worker.worker_id, "task": self_worker.task,
                    "sources": [], "documents": [], "evidences": [], "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_run)

        llm = self._mock_llm()
        svc = ResearchService(llm=llm, checkpoint_store=ck)

        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.resume(task_id)

        assert call_count["run"] == 0, "所有 Worker 已完成，不应再执行"
        assert result["mode"] == "deep"

    def test_resume_reruns_failed_workers(self, monkeypatch, ck):
        """failed Worker 在 Resume 后重新执行"""
        from researchforge.research_service import ResearchService
        call_count = {"W1": 0, "W2": 0}

        orig_run = None

        def mock_run(self_worker):
            call_count[self_worker.worker_id] = call_count.get(self_worker.worker_id, 0) + 1
            return {"worker_id": self_worker.worker_id, "task": self_worker.task,
                    "sources": [], "documents": [], "evidences": [], "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_run)

        task_id = self._prepare_deep_state(ck, ["completed", "failed"])
        llm = self._mock_llm()
        svc = ResearchService(llm=llm, checkpoint_store=ck)

        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.resume(task_id)

        assert call_count.get("W1", 0) == 0, "W1 已完成，不应执行"
        assert call_count.get("W2", 0) == 1, "W2 已失败，应重新执行 1 次"

    def test_resume_reruns_running_workers(self, monkeypatch, ck):
        """running Worker 按未完成处理"""
        from researchforge.research_service import ResearchService
        call_count = {"W1": 0, "W2": 0}

        def mock_run(self_worker):
            call_count[self_worker.worker_id] = call_count.get(self_worker.worker_id, 0) + 1
            return {"worker_id": self_worker.worker_id, "task": self_worker.task,
                    "sources": [], "documents": [], "evidences": [], "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_run)

        task_id = self._prepare_deep_state(ck, ["completed", "running"])
        llm = self._mock_llm()
        svc = ResearchService(llm=llm, checkpoint_store=ck)

        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.resume(task_id)

        assert call_count.get("W1", 0) == 0, "W1 已完成，不应执行"
        assert call_count.get("W2", 0) == 1, "W2 为 running，应重新执行 1 次"

    def test_resume_planning_not_repeated(self, monkeypatch, ck):
        """Planning 不重复执行"""
        from researchforge.research_service import ResearchService
        plan_count = {"plan": 0}

        orig_make_plan = None

        def mock_make_plan(self_lead, topic, num_workers=3):
            plan_count["plan"] += 1
            return ["任务A"]

        monkeypatch.setattr("researchforge.nodes.deep_research.LeadResearcher.make_plan", mock_make_plan)

        task_id = self._prepare_deep_state(ck, ["completed", "completed"])
        llm = self._mock_llm()
        svc = ResearchService(llm=llm, checkpoint_store=ck)

        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.resume(task_id)

        assert plan_count["plan"] == 0, "Planning 不应重复执行"

    def test_resume_partial_merge_correct(self, monkeypatch, ck):
        """部分成功结果正确汇总"""
        from researchforge.research_service import ResearchService
        from researchforge.orchestration import Source

        task_id = self._prepare_deep_state(ck, ["completed", "failed"])
        llm = self._mock_llm()

        def mock_run(self_worker):
            from researchforge.orchestration import Source, Document, Evidence
            return {"worker_id": self_worker.worker_id, "task": self_worker.task,
                    "sources": [Source(id=f"s_{self_worker.worker_id}", title="", snippet="", url="")],
                    "documents": [Document(content=f"d_{self_worker.worker_id}", source_id=f"s_{self_worker.worker_id}")],
                    "evidences": [Evidence(id=f"e_{self_worker.worker_id}", source_id=f"s_{self_worker.worker_id}",
                                            text=f"ev_{self_worker.worker_id}")],
                    "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_run)

        svc = ResearchService(llm=llm, checkpoint_store=ck)
        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.resume(task_id)

        # W1 的结果从 checkpoint 复用，W2 的结果从重跑获得
        # merge 后应有 2 个 sources（W1 旧 + W2 新）
        assert result["stats"]["sources"] >= 2, f"sources 应 >= 2, 实际 {result['stats']['sources']}"

    def test_resume_all_workers_failed(self, monkeypatch, ck):
        """所有 Worker 失败时任务失败"""
        from researchforge.research_service import ResearchService

        call_count = {"W1": 0, "W2": 0}

        def mock_run(self_worker):
            call_count[self_worker.worker_id] = call_count.get(self_worker.worker_id, 0) + 1
            raise RuntimeError(f"{self_worker.worker_id} 持续失败")

        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_run)

        task_id = self._prepare_deep_state(ck, ["failed", "failed"])
        llm = self._mock_llm()
        svc = ResearchService(llm=llm, checkpoint_store=ck)

        with pytest.raises(RuntimeError, match="所有 Worker"):
            with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
                 patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
                 patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
                 patch("researchforge.nodes.audit_node.run_audit_node",
                       return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
                 patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
                svc.resume(task_id)

    def test_resume_api_accepts_deep(self, monkeypatch, ck):
        """Resume API 可以恢复 Deep 任务"""
        from researchforge.research_service import ResearchService

        task_id = self._prepare_deep_state(ck, ["completed", "completed"])
        llm = self._mock_llm()
        svc = ResearchService(llm=llm, checkpoint_store=ck)

        with patch("researchforge.nodes.synthesis_node.run_synthesis_node", return_value=[]), \
             patch("researchforge.nodes.claim_verification_node.run_claim_verification_node", return_value=[]), \
             patch("researchforge.nodes.coverage_node.run_coverage_node", return_value=(True, [])), \
             patch("researchforge.nodes.audit_node.run_audit_node",
                   return_value=type("AuditResult", (), {"passed": True, "issues": [], "suggestions": ""})()), \
             patch("researchforge.nodes.write_node.run_write_node", return_value="报告"):
            result = svc.resume(task_id)

        assert result["mode"] == "deep"
        assert result is not None
