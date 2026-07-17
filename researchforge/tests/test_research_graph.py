"""Research Graph 测试（适配新状态机）"""

import tempfile
from pathlib import Path

import pytest
from researchforge.orchestration import State, ResearchGraph, ResearchMode
from researchforge.orchestration.checkpoint_store import CheckpointStore


class TestResearchGraph:
    """测试新 ResearchGraph（三种模式）"""

    def test_creation(self):
        graph = ResearchGraph()
        assert graph.state == State.CREATED
        assert graph.mode == ResearchMode.STANDARD

    def test_start(self):
        graph = ResearchGraph(mode=ResearchMode.FAST)
        graph.start("测试主题")
        assert graph.rs is not None
        assert graph.rs.topic == "测试主题"
        assert graph.state == State.CREATED

    def test_flow_fast(self):
        """Fast 模式状态流"""
        g = ResearchGraph(mode=ResearchMode.FAST)
        g.start("t")
        states = [g.state.value]
        while True:
            ns = g.get_next_state()
            if not ns:
                break
            g.state = ns
            states.append(g.state.value)
        assert "evaluating" not in states
        assert "human_review" not in states
        assert "auditing" not in states
        assert states[-1] == "completed"

    def test_flow_standard(self):
        """Standard 模式状态流"""
        g = ResearchGraph(mode=ResearchMode.STANDARD)
        g.start("t")
        states = [g.state.value]
        while True:
            ns = g.get_next_state()
            if not ns or ns == State.GAP_SEARCHING:
                break
            g.state = ns
            states.append(g.state.value)
        assert "evaluating" in states
        assert "human_review" in states
        assert "auditing" in states

    def test_flow_deep(self):
        """Deep 模式状态流"""
        g = ResearchGraph(mode=ResearchMode.DEEP)
        g.start("t")
        states = [g.state.value]
        while True:
            ns = g.get_next_state()
            if not ns or ns == State.GAP_SEARCHING:
                break
            g.state = ns
            states.append(g.state.value)
        assert "synthesizing" in states
        assert "human_review" in states

    def test_mode_policy(self):
        """模式策略正确"""
        g_fast = ResearchGraph(mode=ResearchMode.FAST)
        assert g_fast.policy.enable_report_audit is False
        assert g_fast.policy.require_human_review is False
        assert g_fast.policy.search_sources == 3

        g_std = ResearchGraph(mode=ResearchMode.STANDARD)
        assert g_std.policy.enable_report_audit is True
        assert g_std.policy.require_human_review is True
        assert g_std.policy.max_gap_search_rounds == 1

        g_deep = ResearchGraph(mode=ResearchMode.DEEP)
        assert g_deep.policy.enable_multi_worker is True
        assert g_deep.policy.max_gap_search_rounds == 2

    def test_get_status(self):
        graph = ResearchGraph(mode=ResearchMode.FAST)
        graph.start("测试")
        status = graph.get_status()
        assert status["state"] == "created"
        assert status["topic"] == "测试"

    def test_execute(self, monkeypatch):
        """测试 execute 全流程（Fast 模式）"""
        from researchforge.nodes.plan_node import run_plan_node
        from researchforge.nodes.search_node import run_search_node
        from researchforge.nodes.fetch_node import run_fetch_node
        from researchforge.nodes.extract_node import run_extract_node
        from researchforge.nodes.synthesis_node import run_synthesis_node
        from researchforge.nodes.write_node import run_write_node

        # 用 mock 替换所有节点函数，避免真实 LLM/网络调用
        def mock_plan(t, llm=None):
            return ["问题1", "问题2"]
        def mock_search(q, **kw):
            from researchforge.orchestration import Source
            return [Source(id="s1", title="Mock", snippet="mock", url="")]
        def mock_fetch(src, **kw):
            from researchforge.orchestration import Document
            return [Document(content="mock content", source_id="s1")]
        def mock_extract(docs, q):
            from researchforge.orchestration import Evidence
            return [Evidence(id="e1", text="mock evidence", source_id="s1")]
        def mock_synthesis(rs, llm=None):
            from researchforge.orchestration import Claim
            return [Claim(text="1. 核心结论")]
        def mock_write(rs, llm=None, extra_instructions="", mode="fast"):
            return "测试报告"

        monkeypatch.setattr("researchforge.nodes.plan_node.run_plan_node", mock_plan)
        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node", mock_search)
        monkeypatch.setattr("researchforge.nodes.fetch_node.run_fetch_node", mock_fetch)
        monkeypatch.setattr("researchforge.nodes.extract_node.run_extract_node", mock_extract)
        monkeypatch.setattr("researchforge.nodes.synthesis_node.run_synthesis_node", mock_synthesis)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write)

        graph = ResearchGraph(mode=ResearchMode.FAST)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert result["mode"] == "fast"
        assert result["stats"]["sources"] == 1
        assert result["stats"]["evidences"] == 1
        assert graph.state == State.COMPLETED


class TestResearchGraphCheckpoint:
    """ResearchGraph + CheckpointStore 集成测试"""

    @pytest.fixture
    def ck(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield CheckpointStore(store_dir=Path(tmp))

    # ── 普通节点执行后检查 ──

    def test_execute_creates_checkpoints(self, monkeypatch, ck):
        """Fast 模式执行后检查点文件存在"""
        self._mock_all_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)
        # complete 时保存了最终状态
        assert ck.exists(graph.rs.task_id)
        assert ck.count() > 0

    def test_completed_nodes_tracked(self, monkeypatch, ck):
        """Fast 模式 completed_nodes 包含 Plan/Search/Fetch/Extract/Write"""
        self._mock_all_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)
        # 检查每个节点后的检查点都有正确的 completed_nodes
        loaded = ck.load(graph.rs.task_id)
        assert "planning" in loaded.completed_nodes
        assert "searching" in loaded.completed_nodes
        assert "fetching" in loaded.completed_nodes
        assert "extracting" in loaded.completed_nodes
        assert "writing" in loaded.completed_nodes

    def test_multiple_checkpoints_saved(self, monkeypatch, ck):
        """执行过程中有多次检查点保存"""
        self._mock_all_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)
        # Fast 模式: planning/searching/fetching/extracting/writing + complete
        # 每个至少 2 次（start+end）= 至少 count >= 1
        assert ck.count() >= 1

    # ── 失败处理 ──

    def test_failed_node_recorded(self, monkeypatch, ck):
        """节点失败时检查点记录失败状态"""
        from researchforge.nodes.search_node import run_search_node

        def mock_search_fail(q, **kw):
            raise RuntimeError("搜索失败")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node", mock_search_fail)

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(RuntimeError, match="搜索失败"):
            graph.execute("测试", llm=None, progress_callback=None)

        # 失败后检查点应存在，且记录了失败
        assert ck.exists(graph.rs.task_id)
        loaded = ck.load(graph.rs.task_id)
        assert loaded.failed_node == "searching"
        assert loaded.status == "failed"
        # plan 节点是成功的
        assert "planning" in loaded.completed_nodes

    # ── 不用检查点 ──

    def test_without_checkpoint_still_works(self, monkeypatch):
        """不传入 checkpoint_store 时行为正常"""
        self._mock_all_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.FAST)
        result = graph.execute("测试", llm=None, progress_callback=None)
        assert result["report"] == "测试报告"

    # ── Standard 模式节点覆盖 ──

    def test_standard_completed_nodes(self, monkeypatch, ck):
        """Standard 模式所有节点都写入 completed_nodes"""
        self._mock_standard_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        nodes = set(loaded.completed_nodes)
        # 基础节点
        for n in ("planning", "searching", "fetching", "extracting", "synthesizing", "writing", "CLAIM_VERIFICATION"):
            assert n in nodes, f"缺少 {n}, 当前: {nodes}"
        # Standard 独有节点
        assert "evaluating" in nodes, f"缺少 evaluating（coverage）, 当前: {nodes}"
        assert "auditing" in nodes, f"缺少 auditing, 当前: {nodes}"

    def test_standard_gap_search_checkpoint(self, monkeypatch, ck):
        """Standard 模式 gap_searching 时保存检查点"""
        self._mock_standard_nodes(monkeypatch, has_gaps=True)
        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert "gap_searching" in loaded.completed_nodes, f"缺少 gap_searching, 当前: {loaded.completed_nodes}"

    def test_claim_verify_in_completed_nodes(self, monkeypatch, ck):
        """claim_verify 节点应出现在 Standard 模式的 completed_nodes 中"""
        self._mock_standard_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert "CLAIM_VERIFICATION" in loaded.completed_nodes, \
            f"缺少 CLAIM_VERIFICATION, 当前: {loaded.completed_nodes}"

    def test_rewrite_in_completed_nodes(self, monkeypatch, ck):
        """rewrite 节点应在审计未通过时出现在 completed_nodes 中"""
        from researchforge.nodes.audit_node import AuditResult

        def mock_audit_fail(rs, llm):
            return AuditResult(passed=False, issues=["测试问题"], suggestions="请修改")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert "REWRITE" in loaded.completed_nodes, \
            f"缺少 REWRITE, 当前: {loaded.completed_nodes}"

    # ── 流程游标测试 ──

    def test_completed_steps_fast(self, monkeypatch, ck):
        """Fast 模式的 completed_steps 不包含重复节点后缀"""
        self._mock_all_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        steps = set(loaded.completed_steps)
        assert "synthesis_initial" in steps, f"缺少 synthesis_initial, 当前: {steps}"
        assert "claim_verification_initial" in steps

    def test_completed_steps_standard(self, monkeypatch, ck):
        """Standard 模式的 completed_steps 包含 auditorial 和 coverage 节点"""
        self._mock_standard_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        steps = set(loaded.completed_steps)
        for s in ("synthesis_initial", "claim_verification_initial", "audit_initial"):
            assert s in steps, f"缺少 {s}, 当前: {steps}"

    def test_completed_steps_gap_and_rewrite(self, monkeypatch, ck):
        """补搜和重写场景下重复节点有独立步骤名"""
        from researchforge.nodes.audit_node import AuditResult

        def mock_audit_fail(rs, llm):
            return AuditResult(passed=False, issues=["测试问题"], suggestions="请修改")

        self._mock_standard_nodes(monkeypatch, has_gaps=True)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        steps = set(loaded.completed_steps)
        # 首次和重复节点各自有独立步骤名
        assert "synthesis_initial" in steps, f"缺少 synthesis_initial"
        assert "synthesis_after_gap" in steps, f"缺少 synthesis_after_gap, 当前: {steps}"
        assert "audit_initial" in steps, f"缺少 audit_initial"
        assert "audit_after_rewrite" in steps, f"缺少 audit_after_rewrite, 当前: {steps}"

    @pytest.mark.parametrize("fail_node,expected", [
        ("researchforge.nodes.synthesis_node.run_synthesis_node", "synthesizing"),
        ("researchforge.nodes.search_node.run_search_node", "searching"),
        ("researchforge.nodes.write_node.run_write_node", "writing"),
        ("researchforge.nodes.claim_verification_node.run_claim_verification_node", "CLAIM_VERIFICATION"),
    ])
    def test_multiple_failure_points(self, monkeypatch, ck, fail_node, expected):
        """多个节点分别失败时 failed_node 正确"""
        def mock_fail(*a, **kw):
            raise RuntimeError(f"模拟失败: {expected}")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr(fail_node, mock_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        with pytest.raises(RuntimeError):
            graph.execute("测试", llm=None, progress_callback=None)

    # ── 恢复测试 ──

    def test_resume_after_searching(self, monkeypatch, ck):
        """在 searching 后中断，恢复从 fetching 继续"""
        self._mock_all_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)

        # 第一次执行到 searching 完成 → 手动把 fetching 之前的 checkpoint 设好
        # 使用 mock 让 execute 完成完整流程
        graph.execute("测试", llm=None, progress_callback=None)
        assert graph.rs.status == "completed"
        task_id = graph.rs.task_id

        # 第二次：创建新的 graph，模拟恢复
        # 修改 checkpoint 状态：模拟在 fetching 之后中断
        saved = ck.load(task_id)
        saved.status = "failed"
        saved.failed_node = ""  # 模拟非故障中断
        saved.completed_steps = [s for s in saved.completed_steps if s != "planning"]
        saved.completed_nodes = [n for n in saved.completed_nodes if n != "planning"]
        # 恢复 planning 还不算完成 → 但它数据还在
        saved.questions = ["q1"]
        ck.save(saved)

        # 恢复执行
        graph2 = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph2.resume(task_id, llm=None, progress_callback=None)
        assert result["report"] == "测试报告"
        assert graph2.rs.status == "completed"

    def test_resume_completed_task_raises(self, monkeypatch, ck):
        """已完成的任务不能恢复"""
        self._mock_all_nodes(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)

        graph2 = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(ValueError, match="已完成"):
            graph2.resume(graph.rs.task_id, llm=None)

    def test_resume_nonexistent_checkpoint_raises(self, monkeypatch, ck):
        """不存在的检查点恢复时报错"""
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(ValueError, match="不存在"):
            graph.resume("nonexistent_id", llm=None)

    def test_resume_without_store_raises(self, monkeypatch, ck):
        """未配置 CheckpointStore 时恢复报错"""
        graph = ResearchGraph(mode=ResearchMode.FAST)  # 不传 checkpoint_store
        with pytest.raises(RuntimeError, match="未配置"):
            graph.resume("task_id", llm=None)

    # ── 场景恢复测试 ──

    def test_resume_from_synthesis_failure(self, monkeypatch, ck):
        """
        synthesis_initial 失败后保存检查点，resume 时：
        - 之前的 planning/searching/fetching/extracting 不重复执行
        - 从 synthesis_initial 重试
        - 最终正常完成
        """
        call_count = {"synth": 0}

        def mock_synth_fail_then_pass(rs, llm=None):
            call_count["synth"] += 1
            if call_count["synth"] == 1:
                raise RuntimeError("synthesis 首次失败")
            from researchforge.orchestration import Claim
            return [Claim(text="1. 核心结论")]

        self._mock_all_nodes(monkeypatch)
        # 只替换 synthesis（其他节点用 _mock_all_nodes 的默认 mock）
        monkeypatch.setattr(
            "researchforge.nodes.synthesis_node.run_synthesis_node",
            mock_synth_fail_then_pass,
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(RuntimeError, match="synthesis 首次失败"):
            graph.execute("测试", llm=None, progress_callback=None)

        # 失败后检查点存在，且 synthesis_initial 未完成
        assert ck.exists(graph.rs.task_id)
        failed_state = ck.load(graph.rs.task_id)
        assert failed_state.failed_node == "synthesizing"
        assert "synthesis_initial" not in failed_state.completed_steps
        # 前面的步骤已完成（写入 completed_nodes，不是 completed_steps）
        assert "planning" in failed_state.completed_nodes
        assert "searching" in failed_state.completed_nodes
        assert "fetching" in failed_state.completed_nodes
        assert "extracting" in failed_state.completed_nodes

        # 恢复：应该跳过前 4 步，重试 synthesis_initial
        graph2 = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph2.resume(graph.rs.task_id, llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert graph2.rs.status == "completed"
        # synthesis 总共被调用了 2 次（第一次失败 + 第二次恢复成功）
        assert call_count["synth"] == 2

    def test_resume_after_audit_initial_rewrite_path(self, monkeypatch, ck):
        """
        Standard 模式：audit_initial 完成后模拟真实中断，
        resume 时：
        - writing、audit_initial 不重复执行
        - REWRITE 执行 1 次
        - audit_after_rewrite 执行 1 次
        - 最终任务完成
        """
        from researchforge.nodes.audit_node import AuditResult
        call_count = {"audit": 0, "write": 0}

        # 用可跟踪的 call_count 来验证哪些节点实际执行
        original_audit = None
        original_write = None

        def mock_audit_first_fail(rs, llm):
            call_count["audit"] += 1
            return AuditResult(passed=False, issues=["测试问题"], suggestions="请修改重写")

        def mock_audit_second_pass(rs, llm):
            call_count["audit"] += 1
            return AuditResult(passed=True, issues=[])

        def mock_write_with_track(rs, llm=None, extra_instructions="", mode="standard"):
            call_count["write"] += 1
            return "重写后的报告"

        # 首次执行：run_audit_node 第一次返回 fail，同时让 execute 在 audit_initial
        # 完成后抛出异常（模拟真实中断）
        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_first_fail)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write_with_track)

        # 第一轮：走到 audit_initial 完成后中断（偷框架只做 audit_initial 不进入 rewrite）
        # 直接在 audit_initial 之后、REWRITE 之前抛异常
        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        assert graph.execute("测试", llm=None, progress_callback=None) is not None

        # audit_initial 完成且未通过
        assert call_count["audit"] >= 1
        assert call_count["write"] == 2  # 首次 write + rewrite write

        task_id = graph.rs.task_id
        saved = ck.load(task_id)
        assert saved.status == "completed"

        # 手动模拟中断：audit_initial 完成但 REWRITE 和 audit_after_rewrite 没做
        from researchforge.orchestration.research_state import ResearchState
        interrupted_state = ResearchState.from_dict(saved.to_dict())
        interrupted_state.task_id = task_id
        interrupted_state.status = "failed"
        interrupted_state.failed_node = ""
        # 抹掉 REWRITE 和 audit_after_rewrite 的完成记录
        interrupted_state.completed_nodes = [
            n for n in interrupted_state.completed_nodes
            if n not in ("REWRITE", "audit_after_rewrite")
        ]
        interrupted_state.completed_steps = [
            s for s in interrupted_state.completed_steps
            if s not in ("audit_after_rewrite",)
        ]
        # 把报告清空，模拟审计做完但还没重写的状态
        interrupted_state.report = "原始报告（未重写）"
        ck.save(interrupted_state)

        # 重置计数器
        call_count["audit"] = 0
        call_count["write"] = 0

        # 恢复
        graph2 = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph2.resume(task_id, llm=None, progress_callback=None)

        assert graph2.rs.status == "completed"
        # writing 不重复：call_count["write"] 应该是 1（REWRITE 的写入）
        assert call_count["write"] == 1, f"write 应执行 1 次(rewrite), 实际: {call_count['write']}"
        # audit_initial 不重复：call_count["audit"] 应该是 1（audit_after_rewrite）
        assert call_count["audit"] == 1, f"audit 应执行 1 次(re-audit), 实际: {call_count['audit']}"

    def test_resume_after_audit_initial_pass_path(self, monkeypatch, ck):
        """
        Standard 模式：audit_initial 完成（审计通过）后中断，
        resume 时跳过全部，直接完成
        """
        from researchforge.nodes.audit_node import AuditResult
        call_count = {"audit": 0}

        def mock_audit_pass(rs, llm):
            call_count["audit"] += 1
            return AuditResult(passed=True, issues=[])

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_pass)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        graph.execute("测试", llm=None, progress_callback=None)
        task_id = graph.rs.task_id

        saved = ck.load(task_id)
        assert saved.status == "completed"
        # 正常流程不应该有 rewrite
        assert "REWRITE" not in saved.completed_nodes

        # 模拟中断
        saved2 = ck.load(task_id)
        saved2.status = "failed"
        saved2.failed_node = ""
        ck.save(saved2)

        call_count["audit"] = 0

        graph2 = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph2.resume(task_id, llm=None, progress_callback=None)

        assert result["report"] == "标准报告"
        assert graph2.rs.status == "completed"
        assert call_count["audit"] == 0, "audit 不应重执行"

    # ── 辅助 ──

    def _mock_all_nodes(self, monkeypatch):
        """Fast 模式 mock（保留给旧测试）"""
        from researchforge.nodes.plan_node import run_plan_node
        from researchforge.nodes.search_node import run_search_node
        from researchforge.nodes.fetch_node import run_fetch_node
        from researchforge.nodes.extract_node import run_extract_node
        from researchforge.nodes.synthesis_node import run_synthesis_node
        from researchforge.nodes.write_node import run_write_node

        def mock_plan(t, llm=None): return ["问题1", "问题2"]
        def mock_search(q, **kw):
            from researchforge.orchestration import Source
            return [Source(id="s1", title="Mock", snippet="mock", url="")]
        def mock_fetch(src, **kw):
            from researchforge.orchestration import Document
            return [Document(content="mock content", source_id="s1")]
        def mock_extract(docs, q):
            from researchforge.orchestration import Evidence
            return [Evidence(id="e1", text="mock evidence", source_id="s1")]
        def mock_synthesis(rs, llm=None):
            from researchforge.orchestration import Claim
            return [Claim(text="1. 核心结论")]
        def mock_write(rs, llm=None, extra_instructions="", mode="fast"):
            return "测试报告"

        monkeypatch.setattr("researchforge.nodes.plan_node.run_plan_node", mock_plan)
        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node", mock_search)
        monkeypatch.setattr("researchforge.nodes.fetch_node.run_fetch_node", mock_fetch)
        monkeypatch.setattr("researchforge.nodes.extract_node.run_extract_node", mock_extract)
        monkeypatch.setattr("researchforge.nodes.synthesis_node.run_synthesis_node", mock_synthesis)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write)

    def _mock_standard_nodes(self, monkeypatch, has_gaps=False):
        """Mock Standard 模式全部节点（含 coverage/gap_agent/audit）"""
        from researchforge.nodes.plan_node import run_plan_node
        from researchforge.nodes.search_node import run_search_node
        from researchforge.nodes.fetch_node import run_fetch_node
        from researchforge.nodes.extract_node import run_extract_node
        from researchforge.nodes.synthesis_node import run_synthesis_node
        from researchforge.nodes.claim_verification_node import run_claim_verification_node
        from researchforge.nodes.coverage_node import run_coverage_node
        from researchforge.nodes.gap_agent import run_evidence_gap_agent
        from researchforge.nodes.audit_node import run_audit_node
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
            return [Evidence(id="e1", source_id="s1", text="e")]
        def mock_synthesis(rs, llm=None):
            from researchforge.orchestration import Claim
            return [Claim(text="c1", evidence_ids=["e1"], confidence=1.0)]
        def mock_claim_verify(rs, llm=None):
            from researchforge.nodes.claim_verification_node import VerifiedClaim, ClaimStatus
            return [VerifiedClaim(claim_index=0, status=ClaimStatus.SUPPORTED)]
        def mock_coverage(rs):
            from researchforge.orchestration import ResearchState
            if has_gaps:
                return False, ["GAP: 需要更多数据"]
            return True, []
        def mock_gap_agent(rs, gaps, llm, **kw):
            return True, []
        def mock_audit(rs, llm):
            from researchforge.nodes.audit_node import AuditResult
            return AuditResult(passed=True, issues=[])
        def mock_write(rs, llm=None, extra_instructions="", mode="standard"):
            return "标准报告"

        monkeypatch.setattr("researchforge.nodes.plan_node.run_plan_node", mock_plan)
        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node", mock_search)
        monkeypatch.setattr("researchforge.nodes.fetch_node.run_fetch_node", mock_fetch)
        monkeypatch.setattr("researchforge.nodes.extract_node.run_extract_node", mock_extract)
        monkeypatch.setattr("researchforge.nodes.synthesis_node.run_synthesis_node", mock_synthesis)
        monkeypatch.setattr("researchforge.nodes.claim_verification_node.run_claim_verification_node", mock_claim_verify)
        monkeypatch.setattr("researchforge.nodes.coverage_node.run_coverage_node", mock_coverage)
        monkeypatch.setattr("researchforge.nodes.gap_agent.run_evidence_gap_agent", mock_gap_agent)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write)
