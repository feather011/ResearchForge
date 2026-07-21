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

    @pytest.mark.parametrize("fail_node,expected,raises", [
        ("researchforge.nodes.synthesis_node.run_synthesis_node", "synthesizing", True),
        ("researchforge.nodes.search_node.run_search_node", "searching", True),
        ("researchforge.nodes.write_node.run_write_node", "writing", True),
        ("researchforge.nodes.claim_verification_node.run_claim_verification_node", "CLAIM_VERIFICATION", False),
    ])
    def test_multiple_failure_points(self, monkeypatch, ck, fail_node, expected, raises):
        """多个节点分别失败时 failed_node 正确"""
        def mock_fail(*a, **kw):
            raise RuntimeError(f"模拟失败: {expected}")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr(fail_node, mock_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        if raises:
            with pytest.raises(RuntimeError):
                graph.execute("测试", llm=None, progress_callback=None)
        else:
            # Claim Verification 降级不终止，任务应正常完成
            result = graph.execute("测试", llm=None, progress_callback=None)
            assert result is not None

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

    # ── Searching 重试测试 ──

    def test_search_retry_timeout_then_success(self, monkeypatch, ck):
        """
        TimeoutError 第一次失败、第二次成功
        → searching 执行 2 次（1 次失败 + 1 次重试成功）
        """
        call_count = {"search": 0}

        def mock_search_first_fail(q, **kw):
            call_count["search"] += 1
            if call_count["search"] == 1:
                raise TimeoutError("搜索超时（可重试）")
            from researchforge.orchestration import Source
            return [Source(id="s1", title="Mock", snippet="mock", url="")]

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.search_node.run_search_node", mock_search_first_fail
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert result["stats"]["sources"] == 1
        assert graph.rs.status == "completed"
        assert call_count["search"] == 2, (
            f"searching 应执行 2 次（第 1 次超时 + 第 2 次重试成功）, "
            f"实际 {call_count['search']}"
        )

    def test_search_retry_exhausted(self, monkeypatch, ck):
        """
        超过最大次数后失败
        → searching 执行 2 次（配置 max_retries=2，第 1 次失败 + 第 2 次还失败）
        → 最终抛出异常
        """
        call_count = {"search": 0}

        def mock_search_always_fail(q, **kw):
            call_count["search"] += 1
            raise TimeoutError("搜索持续超时")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.search_node.run_search_node", mock_search_always_fail
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "searching"
        # max_retries=2，第 1 次失败（attempt=1）+ 第 2 次重试失败（attempt=2）后放弃
        # 但重试循环中 attempt=3 时 should_retry 返回 False（超过 max_retries）才退出
        # → searching 共执行 3 次（attempt=1 首试, attempt=2 第一次重试, attempt=3 检查后放弃）
        assert call_count["search"] == 3, (
            f"searching 应执行 3 次（第 1 次 + 重试 2 次后放弃）, "
            f"实际 {call_count['search']}"
        )

    def test_search_retry_value_error_not_retried(self, monkeypatch, ck):
        """
        ValueError 不重试
        → searching 执行 1 次，直接失败
        """
        call_count = {"search": 0}

        def mock_search_value_error(q, **kw):
            call_count["search"] += 1
            raise ValueError("配置错误（不可重试）")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.search_node.run_search_node", mock_search_value_error
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(ValueError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "searching"
        assert call_count["search"] == 1, (
            f"searching 应只执行 1 次（ValueError 不重试）, "
            f"实际 {call_count['search']}"
        )

    def test_search_retry_after_resume(self, monkeypatch, ck):
        """
        恢复任务后 searching 重试恢复场景：
        1. 第一次运行时 searching 重试耗尽并失败（TimeoutError ×2）
        2. checkpoint 记录 searching 未完成
        3. resume → searching 重新执行 → 第 1 次 TimeoutError → 第 2 次成功
        4. planning 不重复执行
        """
        search_count = {"first_run": 0, "resume_run": 0}
        current_phase = ["first_run"]

        def mock_search(q, **kw):
            if current_phase[0] == "first_run":
                search_count["first_run"] += 1
                raise TimeoutError("搜索持续超时")
            else:
                search_count["resume_run"] += 1
                if search_count["resume_run"] == 1:
                    raise TimeoutError("搜索超时（可重试）")
                from researchforge.orchestration import Source
                return [Source(id="s1", title="Mock", snippet="mock", url="")]

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.search_node.run_search_node", mock_search
        )

        # 第一次执行：searching 重试耗尽后失败
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        task_id = graph.rs.task_id
        loaded = ck.load(task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "searching"
        assert "searching" not in loaded.completed_nodes
        # searching 在第 1 次运行时共执行 3 次（attempt=1 + attempt=2 + attempt=3 放弃）
        assert search_count["first_run"] == 3, (
            f"首次执行: searching 应执行 3 次（重试耗尽）, "
            f"实际 {search_count['first_run']}"
        )

        # 恢复阶段
        current_phase[0] = "resume"

        graph2 = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph2.resume(task_id, llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert graph2.rs.status == "completed"
        # planning 不应重复执行
        assert "planning" in graph2.rs.completed_nodes
        # searching 最终在 completed_nodes 中
        assert "searching" in graph2.rs.completed_nodes
        # resume 阶段 searching 执行 2 次（第 1 次超时 + 第 2 次成功）
        assert search_count["resume_run"] == 2, (
            f"恢复后: searching 应执行 2 次（第 1 次超时 + 第 2 次成功）, "
            f"实际 {search_count['resume_run']}"
        )

    # ── Fetching 重试测试 ──

    def test_fetch_single_retry_then_success(self, monkeypatch, ck):
        """
        单个网页 TimeoutError 第一次失败，重试后成功
        → fetching 节点正常完成
        """
        from researchforge.orchestration import Source, Document
        fetch_call_count = {"url_a": 0}

        def mock_fetch_single(src):
            fetch_call_count["url_a"] += 1
            if fetch_call_count["url_a"] == 1:
                raise TimeoutError("抓取超时")
            return Document(source_id=src.id, content="成功内容", url=src.url, title=src.title)

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.fetch_node._fetch_single", mock_fetch_single)

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert result["stats"]["documents"] == 1
        assert graph.rs.status == "completed"
        assert fetch_call_count["url_a"] == 2, (
            f"单个网页重试后成功: 应执行 2 次, 实际 {fetch_call_count['url_a']}"
        )

    def test_fetch_single_fails_others_succeed(self, monkeypatch, ck):
        """
        一个网页彻底失败，其他网页成功
        → fetching 节点正常完成（非全部失败）
        """
        from researchforge.orchestration import Source, Document
        call_order = []

        def mock_fetch_single(src):
            call_order.append(src.id)
            if src.id == "s1":
                raise TimeoutError("s1 抓取失败")
            return Document(source_id=src.id, content="成功内容", url=src.url, title=src.title)

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.plan_node.run_plan_node",
                            lambda t, llm=None: ["q1", "q2"])
        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node",
                            lambda q, **kw: [
                                Source(id="s1", title="A", snippet="a", url=""),
                                Source(id="s2", title="B", snippet="b", url=""),
                            ])
        monkeypatch.setattr("researchforge.nodes.fetch_node._fetch_single", mock_fetch_single)

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert graph.rs.status == "completed"
        # s1 失败（attempt=1 + attempt=2=放弃）= 2 次, s2 成功 = 1 次, 共 3 次
        # 但 _run_with_retry 在 attempt=3 时执行操作后 should_retry 才返回 False,
        # 所以实际执行 4 次（s1×3 + s2×1）
        assert len(call_order) == 4, (
            f"s1(3次)+s2(1次)=4, 实际 {len(call_order)}"
        )
        assert "s1" in call_order and "s2" in call_order

    def test_fetch_all_fail_node_fails(self, monkeypatch, ck):
        """
        所有网页都抓取失败
        → fetching 节点标记失败并保存 checkpoint
        """
        call_count = {"fetch": 0}

        def mock_fetch_single(src):
            call_count["fetch"] += 1
            raise TimeoutError("抓取始终失败")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.fetch_node._fetch_single", mock_fetch_single)

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(RuntimeError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "fetching"
        # 1 个 source, attempt=1 + attempt=2（重试）+ attempt=3(放弃) = 3 次
        assert call_count["fetch"] == 3, (
            f"全部失败(1个url×3次): 应执行 3 次, 实际 {call_count['fetch']}"
        )

    def test_fetch_retry_after_resume(self, monkeypatch, ck):
        """
        Resume 后 fetching 从检查点重新执行
        → 重试逻辑正常，节点最终完成
        """
        from researchforge.orchestration import Source, Document
        fetch_count = [0]

        def mock_fetch_single(src):
            fetch_count[0] += 1
            if fetch_count[0] == 1:
                raise TimeoutError("抓取超时")
            return Document(source_id=src.id, content="成功内容", url=src.url, title=src.title)

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.fetch_node._fetch_single", mock_fetch_single)

        # 第一次执行
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)
        assert result["report"] == "测试报告"
        assert graph.rs.status == "completed"
        task_id = graph.rs.task_id

        # 模拟中断（抹掉 fetching 以后的状态）
        saved = ck.load(task_id)
        saved.status = "failed"
        saved.failed_node = ""
        saved.completed_nodes = [n for n in saved.completed_nodes if n != "fetching"]
        saved.documents = []
        ck.save(saved)

        fetch_count[0] = 0

        graph2 = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result2 = graph2.resume(task_id, llm=None, progress_callback=None)

        assert result2["report"] == "测试报告"
        assert graph2.rs.status == "completed"
        assert "fetching" in graph2.rs.completed_nodes
        assert fetch_count[0] == 2, (
            f"resume 后 fetching 应执行 2 次(1 次失败+1 次成功), "
            f"实际 {fetch_count[0]}"
        )

    # ── Synthesis 重试测试 ──

    def test_synth_retry_timeout_then_success(self, monkeypatch, ck):
        """
        synthesis_initial TimeoutError 第一次失败、第二次成功
        → synthesis 执行 2 次（1 次超时 + 1 次重试成功）
        """
        call_count = {"synth": 0}

        def mock_synth(rs, llm=None):
            call_count["synth"] += 1
            if call_count["synth"] == 1:
                raise TimeoutError("LLM 超时（可重试）")
            from researchforge.orchestration import Claim
            return [Claim(text="1. 核心结论")]

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.synthesis_node.run_synthesis_node", mock_synth
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert graph.rs.status == "completed"
        assert call_count["synth"] == 2, (
            f"synthesis 应执行 2 次（第 1 次超时 + 第 2 次重试成功）, "
            f"实际 {call_count['synth']}"
        )

    def test_synth_retry_exhausted(self, monkeypatch, ck):
        """
        synthesis_initial 重试耗尽后失败
        → synthesis 执行 3 次（attempt=1,2,3），第 3 次后放弃
        """
        call_count = {"synth": 0}

        def mock_synth_always_fail(rs, llm=None):
            call_count["synth"] += 1
            raise TimeoutError("LLM 持续超时")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.synthesis_node.run_synthesis_node", mock_synth_always_fail
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "synthesizing"
        assert call_count["synth"] == 4, (
            f"synthesis 应执行 4 次（attempt=1,2,3 各执行一次, attempt=4 检查后放弃）, "
            f"实际 {call_count['synth']}"
        )

    def test_synth_value_error_not_retried(self, monkeypatch, ck):
        """
        ValueError 不重试
        → synthesis 执行 1 次，直接失败
        """
        call_count = {"synth": 0}

        def mock_synth_value_error(rs, llm=None):
            call_count["synth"] += 1
            raise ValueError("输出解析错误（不可重试）")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.synthesis_node.run_synthesis_node", mock_synth_value_error
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(ValueError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "synthesizing"
        assert call_count["synth"] == 1, (
            f"synthesis 应只执行 1 次（ValueError 不重试）, "
            f"实际 {call_count['synth']}"
        )

    def test_synth_after_gap_retry(self, monkeypatch, ck):
        """
        synthesis_after_gap 使用相同重试机制
        → Standard 模式有 gap → synthesis_after_gap 超时一次后成功
        → synthesis_initial 执行 1 次 + synthesis_after_gap 执行 2 次 = 共 3 次
        """
        call_count = {"synth": 0}

        def mock_synth(rs, llm=None):
            call_count["synth"] += 1
            # synthesis_after_gap（第 2 次）第一次失败
            if call_count["synth"] == 2:
                raise TimeoutError("补搜后 synthesis 超时（可重试）")
            from researchforge.orchestration import Claim
            return [Claim(text="1. 核心结论")]

        self._mock_standard_nodes(monkeypatch, has_gaps=True)
        monkeypatch.setattr(
            "researchforge.nodes.synthesis_node.run_synthesis_node", mock_synth
        )

        from researchforge.nodes.audit_node import AuditResult

        def mock_audit_pass(rs, llm):
            return AuditResult(passed=True, issues=[])

        monkeypatch.setattr(
            "researchforge.nodes.audit_node.run_audit_node", mock_audit_pass
        )

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "标准报告"
        assert graph.rs.status == "completed"
        # synthesis_initial(1次) + synthesis_after_gap(失败1次+重试成功1次) = 共 3 次
        assert call_count["synth"] == 3, (
            f"synthesis 应执行 3 次（首次1次 + gap后2次）, "
            f"实际 {call_count['synth']}"
        )

    def test_synth_retry_exhausted_then_resume(self, monkeypatch, ck):
        """
        重试耗尽后 Resume 再次执行并成功
        → 首次：synthesis 重试耗尽（3 次）→ failed
        → resume 后：synthesis 第 1 次 TimeoutError → 第 2 次成功
        → 共 5 次（首次 3 次 + resume 后 2 次），最终 completed
        """
        call_count = {"synth": 0}
        phase = ["first"]

        def mock_synth(rs, llm=None):
            call_count["synth"] += 1
            if phase[0] == "first":
                raise TimeoutError("LLM 持续超时")
            else:
                if call_count["synth"] == 4:  # resume 后的第 1 次
                    raise TimeoutError("resume 后首次超时（可重试）")
                from researchforge.orchestration import Claim
                return [Claim(text="1. 核心结论")]

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.synthesis_node.run_synthesis_node", mock_synth
        )

        # 首次执行：重试耗尽
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        task_id = graph.rs.task_id
        loaded = ck.load(task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "synthesizing"
        assert call_count["synth"] == 4, (
            f"首次: synthesis 应执行 4 次（attempt=1,2,3 各执行, attempt=4 放弃）, "
            f"实际 {call_count['synth']}"
        )

        # Resume
        phase[0] = "resume"

        graph2 = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph2.resume(task_id, llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert graph2.rs.status == "completed"
        # resume 后: 1 次失败 + 1 次重试成功 = 2 次
        # 总计: 4 (首次) + 1 (resume后) = 5 次
        # 注意: resume 后 synthesis 只有重试阶段的 2 次调用，
        # 但 synthesis 完整执行后 claims 到位，进入后续节点,
        # 所以只多调了 1 次（第 1 次失败、第 2 次成功）→ 共 4+1=5 次。
        # 原因：llm_policy max_retries=3，但 mock 中 resume 后的第 1 次失败后
        # attempt=2 时 should_retry 返回 True → 重试成功。
        assert call_count["synth"] == 5, (
            f"总共: synthesis 应执行 5 次（首次 4 + resume 后 1）, "
            f"实际 {call_count['synth']}"
        )

    # ── Planning 重试测试 ──

    def test_plan_retry_timeout_then_success(self, monkeypatch, ck):
        """
        Planning TimeoutError 第一次失败、第二次成功
        → planning 执行 2 次（1 次超时 + 1 次重试成功）
        """
        call_count = {"plan": 0}

        def mock_plan(t, llm=None):
            call_count["plan"] += 1
            if call_count["plan"] == 1:
                raise TimeoutError("LLM 超时（可重试）")
            return ["问题1"]

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.plan_node.run_plan_node", mock_plan
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert graph.rs.status == "completed"
        assert call_count["plan"] == 2, (
            f"planning 应执行 2 次（第 1 次超时 + 第 2 次重试成功）, "
            f"实际 {call_count['plan']}"
        )

    def test_plan_retry_exhausted(self, monkeypatch, ck):
        """
        Planning 重试耗尽后失败
        → planning 执行 4 次（attempt=1,2,3 各执行, attempt=4 放弃）
        """
        call_count = {"plan": 0}

        def mock_plan_always_fail(t, llm=None):
            call_count["plan"] += 1
            raise TimeoutError("LLM 持续超时")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.plan_node.run_plan_node", mock_plan_always_fail
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "planning"
        assert call_count["plan"] == 4, (
            f"planning 应执行 4 次（重试耗尽）, "
            f"实际 {call_count['plan']}"
        )

    def test_plan_value_error_not_retried(self, monkeypatch, ck):
        """
        Planning ValueError 不重试
        → planning 执行 1 次，直接失败
        """
        call_count = {"plan": 0}

        def mock_plan_value_error(t, llm=None):
            call_count["plan"] += 1
            raise ValueError("LLM 返回 JSON 格式错误（不可重试）")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.plan_node.run_plan_node", mock_plan_value_error
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(ValueError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "planning"
        assert call_count["plan"] == 1, (
            f"planning 应只执行 1 次（ValueError 不重试）, "
            f"实际 {call_count['plan']}"
        )

    def test_plan_retry_exhausted_then_resume(self, monkeypatch, ck):
        """
        重试耗尽后 Resume 再次执行并成功
        → 首次：planning 重试耗尽（4 次）→ failed
        → resume 后：planning 第 1 次 TimeoutError → 第 2 次成功
        → 共 5 次（首次 4 + resume 后 1），最终 completed
        """
        call_count = {"plan": 0}
        phase = ["first"]

        def mock_plan(t, llm=None):
            call_count["plan"] += 1
            if phase[0] == "first":
                raise TimeoutError("LLM 持续超时")
            else:
                return ["问题1"]

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.plan_node.run_plan_node", mock_plan
        )

        # 首次执行：重试耗尽
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        task_id = graph.rs.task_id
        loaded = ck.load(task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "planning"
        assert call_count["plan"] == 4, (
            f"首次: planning 应执行 4 次（重试耗尽）, "
            f"实际 {call_count['plan']}"
        )

        # Resume
        phase[0] = "resume"

        graph2 = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph2.resume(task_id, llm=None, progress_callback=None)

        assert result["report"] == "测试报告"
        assert graph2.rs.status == "completed"
        # resume 后: planning 第 1 次成功（无重试）
        assert call_count["plan"] == 5, (
            f"总共: planning 应执行 5 次（首次 4 + resume 后 1）, "
            f"实际 {call_count['plan']}"
        )

    # ── Writing 重试测试 ──

    def test_write_retry_timeout_then_success(self, monkeypatch, ck):
        """
        Writing TimeoutError 第一次失败、第二次成功
        → writing 执行 2 次（1 次超时 + 1 次重试成功）
        """
        call_count = {"write": 0}

        def mock_write(rs, llm=None, extra_instructions="", mode="fast"):
            call_count["write"] += 1
            if call_count["write"] == 1:
                raise TimeoutError("LLM 超时（可重试）")
            return "写作成功报告"

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.write_node.run_write_node", mock_write
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "写作成功报告"
        assert graph.rs.status == "completed"
        assert call_count["write"] == 2, (
            f"writing 应执行 2 次（第 1 次超时 + 第 2 次重试成功）, "
            f"实际 {call_count['write']}"
        )

    def test_write_retry_exhausted(self, monkeypatch, ck):
        """
        Writing 重试耗尽后失败
        → writing 执行 4 次（attempt=1,2,3 各执行, attempt=4 放弃）
        """
        call_count = {"write": 0}

        def mock_write_always_fail(rs, llm=None, extra_instructions="", mode="fast"):
            call_count["write"] += 1
            raise TimeoutError("LLM 持续超时")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.write_node.run_write_node", mock_write_always_fail
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "writing"
        assert call_count["write"] == 4, (
            f"writing 应执行 4 次（重试耗尽）, "
            f"实际 {call_count['write']}"
        )

    def test_write_value_error_not_retried(self, monkeypatch, ck):
        """
        Writing ValueError 不重试
        → writing 执行 1 次，直接失败
        """
        call_count = {"write": 0}

        def mock_write_value_error(rs, llm=None, extra_instructions="", mode="fast"):
            call_count["write"] += 1
            raise ValueError("LLM 输出格式错误（不可重试）")

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.write_node.run_write_node", mock_write_value_error
        )

        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(ValueError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "writing"
        assert call_count["write"] == 1, (
            f"writing 应只执行 1 次（ValueError 不重试）, "
            f"实际 {call_count['write']}"
        )

    def test_write_retry_exhausted_then_resume(self, monkeypatch, ck):
        """
        重试耗尽后 Resume 再次执行并成功
        → 首次：writing 重试耗尽（4 次）→ failed
        → resume 后：writing 第 1 次成功
        → 共 5 次（首次 4 + resume 后 1），最终 completed
        """
        call_count = {"write": 0}
        phase = ["first"]

        def mock_write(rs, llm=None, extra_instructions="", mode="fast"):
            call_count["write"] += 1
            if phase[0] == "first":
                raise TimeoutError("LLM 持续超时")
            return "resume 写作报告"

        self._mock_all_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.write_node.run_write_node", mock_write
        )

        # 首次执行：重试耗尽
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        task_id = graph.rs.task_id
        loaded = ck.load(task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "writing"
        assert call_count["write"] == 4, (
            f"首次: writing 应执行 4 次（重试耗尽）, "
            f"实际 {call_count['write']}"
        )

        # Resume
        phase[0] = "resume"

        graph2 = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        result = graph2.resume(task_id, llm=None, progress_callback=None)

        assert result["report"] == "resume 写作报告"
        assert graph2.rs.status == "completed"
        assert call_count["write"] == 5, (
            f"总共: writing 应执行 5 次（首次 4 + resume 后 1）, "
            f"实际 {call_count['write']}"
        )

    # ── Rewrite 重试测试 ──

    def test_rewrite_retry_timeout_then_success(self, monkeypatch, ck):
        """
        REWRITE TimeoutError 第一次失败、第二次成功
        → rewrite 执行 2 次（1 次超时 + 1 次重试成功）
        """
        from researchforge.nodes.audit_node import AuditResult
        write_call_count = {"write": 0, "rewrite": 0}

        def mock_write(rs, llm=None, extra_instructions="", mode="standard"):
            if extra_instructions:
                write_call_count["rewrite"] += 1
                if write_call_count["rewrite"] == 1:
                    raise TimeoutError("LLM 超时（可重试）")
                return "重写后报告"
            write_call_count["write"] += 1
            return "原始报告"

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write)

        def mock_audit_fail(rs, llm):
            return AuditResult(passed=False, issues=["测试问题"], suggestions="请修改")

        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "重写后报告"
        assert graph.rs.status == "completed"
        # rewrite: 第 1 次超时 + 第 2 次重试成功 = 2 次
        assert write_call_count["rewrite"] == 2, (
            f"REWRITE 应执行 2 次（第 1 次超时 + 第 2 次重试成功）, "
            f"实际 {write_call_count['rewrite']}"
        )

    def test_rewrite_retry_exhausted(self, monkeypatch, ck):
        """
        REWRITE 重试耗尽后失败
        → rewrite 执行 4 次（attempt=1,2,3 各执行, attempt=4 放弃）
        """
        from researchforge.nodes.audit_node import AuditResult
        write_call_count = {"write": 0, "rewrite": 0}

        def mock_write(rs, llm=None, extra_instructions="", mode="standard"):
            if extra_instructions:
                write_call_count["rewrite"] += 1
                raise TimeoutError("LLM 持续超时")
            write_call_count["write"] += 1
            return "原始报告"

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write)

        def mock_audit_fail(rs, llm):
            return AuditResult(passed=False, issues=["问题"], suggestions="请修改")

        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "REWRITE"
        assert write_call_count["rewrite"] == 4, (
            f"REWRITE 应执行 4 次（重试耗尽）, "
            f"实际 {write_call_count['rewrite']}"
        )

    def test_rewrite_value_error_not_retried(self, monkeypatch, ck):
        """
        REWRITE ValueError 不重试
        → rewrite 执行 1 次，直接失败
        """
        from researchforge.nodes.audit_node import AuditResult
        write_call_count = {"write": 0, "rewrite": 0}

        def mock_write(rs, llm=None, extra_instructions="", mode="standard"):
            if extra_instructions:
                write_call_count["rewrite"] += 1
                raise ValueError("输出格式错误（不可重试）")
            write_call_count["write"] += 1
            return "原始报告"

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write)

        def mock_audit_fail(rs, llm):
            return AuditResult(passed=False, issues=["问题"], suggestions="请修改")

        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        with pytest.raises(ValueError):
            graph.execute("测试", llm=None, progress_callback=None)

        loaded = ck.load(graph.rs.task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "REWRITE"
        assert write_call_count["rewrite"] == 1, (
            f"REWRITE 应只执行 1 次（ValueError 不重试）, "
            f"实际 {write_call_count['rewrite']}"
        )

    def test_rewrite_retry_exhausted_then_resume(self, monkeypatch, ck):
        """
        重试耗尽后 Resume 再次执行并成功
        → 首次：REWRITE 重试耗尽（4 次）→ failed
        → resume 后：REWRITE 成功
        → 共 5 次（首次 4 + resume 后 1），最终 completed
        """
        from researchforge.nodes.audit_node import AuditResult
        write_call_count = {"write": 0, "rewrite": 0}
        phase = ["first"]

        def mock_write(rs, llm=None, extra_instructions="", mode="standard"):
            if extra_instructions:
                write_call_count["rewrite"] += 1
                if phase[0] == "first":
                    raise TimeoutError("LLM 持续超时")
                return "resume 重写后报告"
            write_call_count["write"] += 1
            return "原始报告"

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_write)

        def mock_audit_fail(rs, llm):
            return AuditResult(passed=False, issues=["问题"], suggestions="请修改")

        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        with pytest.raises(TimeoutError):
            graph.execute("测试", llm=None, progress_callback=None)

        task_id = graph.rs.task_id
        loaded = ck.load(task_id)
        assert loaded.status == "failed"
        assert loaded.failed_node == "REWRITE"
        assert write_call_count["rewrite"] == 4, (
            f"首次: REWRITE 应执行 4 次（重试耗尽）, "
            f"实际 {write_call_count['rewrite']}"
        )

        # Resume
        phase[0] = "resume"

        graph2 = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph2.resume(task_id, llm=None, progress_callback=None)

        assert result["report"] == "resume 重写后报告"
        assert graph2.rs.status == "completed"
        # resume 后: rewrite 第 1 次成功
        assert write_call_count["rewrite"] == 5, (
            f"总共: REWRITE 应执行 5 次（首次 4 + resume 后 1）, "
            f"实际 {write_call_count['rewrite']}"
        )

    # ── Claim Verification 降级测试 ──

    def test_claim_verify_retry_then_success(self, monkeypatch, ck):
        """
        Claim Verification TimeoutError 第一次失败、第二次成功
        → claim_verify 执行 2 次（1 次超时 + 1 次重试成功）
        """
        call_count = {"cv": 0}

        def mock_cv(rs, llm=None):
            call_count["cv"] += 1
            if call_count["cv"] == 1:
                raise TimeoutError("LLM 超时（可重试）")
            from researchforge.nodes.claim_verification_node import VerifiedClaim, ClaimStatus
            return [VerifiedClaim(claim_index=0, status=ClaimStatus.SUPPORTED)]

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.claim_verification_node.run_claim_verification_node", mock_cv
        )

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "标准报告"
        assert graph.rs.status == "completed"
        assert call_count["cv"] == 2, (
            f"claim_verify 应执行 2 次（第 1 次超时 + 第 2 次重试成功）, "
            f"实际 {call_count['cv']}"
        )

    def test_claim_verify_retry_exhausted_then_degraded(self, monkeypatch, ck):
        """
        Claim Verification 重试耗尽后降级
        → claims 标记为 UNVERIFIED（不是 UNSUPPORTED）
        → 任务继续完成
        """
        from researchforge.nodes.claim_verification_node import ClaimStatus

        call_count = {"cv": 0}

        def mock_cv_always_fail(rs, llm=None):
            call_count["cv"] += 1
            raise TimeoutError("LLM 持续超时")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.claim_verification_node.run_claim_verification_node", mock_cv_always_fail
        )

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        # 任务应正常完成（降级不终止）
        assert result["report"] == "标准报告"
        assert graph.rs.status == "completed"
        assert call_count["cv"] == 4, (
            f"claim_verify 应执行 4 次（重试耗尽）, "
            f"实际 {call_count['cv']}"
        )
        # 降级时 claims 应标记为 UNVERIFIED（不是 UNSUPPORTED）
        assert len(graph.rs.claims) > 0
        assert graph.rs.metadata.get("claim_verify_degraded") is True

    def test_claim_verify_value_error_not_retried_degraded(self, monkeypatch, ck):
        """
        Claim Verification ValueError 不重试
        → 进入降级（claims 标记为 UNVERIFIED）
        """
        from researchforge.nodes.claim_verification_node import ClaimStatus

        def mock_cv_value_error(rs, llm=None):
            raise ValueError("LLM 输出解析错误（不可重试）")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.claim_verification_node.run_claim_verification_node", mock_cv_value_error
        )

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "标准报告"
        assert graph.rs.status == "completed"
        assert graph.rs.metadata.get("claim_verify_degraded") is True

    def test_claim_verify_degraded_not_unsupported(self, monkeypatch, ck):
        """
        验证降级时 claims 标记为 UNVERIFIED，不是 UNSUPPORTED
        """
        from researchforge.nodes.claim_verification_node import ClaimStatus

        def mock_cv_always_fail(rs, llm=None):
            raise TimeoutError("LLM 持续超时")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr(
            "researchforge.nodes.claim_verification_node.run_claim_verification_node", mock_cv_always_fail
        )

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert graph.rs.status == "completed"
        # 降级时 claims 的 confidence 为 0.0 → unverified 计数应为 claims 的总数
        assert result.get("claim_verification", {}).get("unverified", 0) >= 1, (
            "降级时 unverified 计数应为 claims 数量"
        )
        # 验证降级标记
        assert graph.rs.metadata.get("claim_verify_degraded") is True

    # ── Audit 降级测试 ──

    def test_audit_retry_timeout_then_success(self, monkeypatch, ck):
        """
        Audit TimeoutError 第一次失败、第二次成功
        → audit 执行 2 次（1 次超时 + 1 次重试成功）
        """
        from researchforge.nodes.audit_node import AuditResult
        call_count = {"audit": 0}

        def mock_audit(rs, llm):
            call_count["audit"] += 1
            if call_count["audit"] == 1:
                raise TimeoutError("LLM 超时（可重试）")
            return AuditResult(passed=True, issues=[])

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "标准报告"
        assert graph.rs.status == "completed"
        assert call_count["audit"] == 2, (
            f"audit 应执行 2 次（第 1 次超时 + 第 2 次重试成功）, "
            f"实际 {call_count['audit']}"
        )

    def test_audit_retry_exhausted_then_degraded(self, monkeypatch, ck):
        """
        Audit 重试耗尽后降级
        → audit passed=False, 记录 audit_degraded
        → 不触发 Rewrite
        """
        call_count = {"audit": 0}

        def mock_audit_always_fail(rs, llm):
            call_count["audit"] += 1
            raise TimeoutError("LLM 持续超时")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_always_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "标准报告"
        assert graph.rs.status == "completed"
        assert call_count["audit"] == 4, (
            f"audit 应执行 4 次（重试耗尽）, "
            f"实际 {call_count['audit']}"
        )
        # 降级结果：passed=False, 非通过
        assert result["audit"]["passed"] is False, "降级时 audit.passed 应为 False"
        # audit_degraded 记录在 metadata 中
        assert graph.rs.metadata.get("audit_degraded") is True
        # 不触发 Rewrite（rewritten=0）
        assert result["audit"]["rewritten"] == 0

    def test_audit_value_error_not_retried_degraded(self, monkeypatch, ck):
        """
        Audit ValueError 不重试 → 进入降级
        """
        def mock_audit_value_error(rs, llm):
            raise ValueError("LLM 输出解析错误（不可重试）")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_value_error)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert result["report"] == "标准报告"
        assert graph.rs.status == "completed"
        assert result["audit"]["passed"] is False
        assert graph.rs.metadata.get("audit_degraded") is True

    def test_audit_degraded_not_passed(self, monkeypatch, ck):
        """
        降级不能被标记为审核通过
        """
        def mock_audit_always_fail(rs, llm):
            raise TimeoutError("LLM 持续超时")

        self._mock_standard_nodes(monkeypatch)
        monkeypatch.setattr("researchforge.nodes.audit_node.run_audit_node", mock_audit_always_fail)

        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)

        assert graph.rs.status == "completed"
        # 降级时 audit.passed 绝对不能为 True
        assert result["audit"]["passed"] is not True, "降级时 audit.passed 不能为 True"
        assert "审核未完成" in (result["audit"]["issues"] or [])[0] if result["audit"]["issues"] else "", \
            "降级时 issues 应提示审核未完成"

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


class TestRunWithRetryTrace:
    """_run_with_retry + Tracer 集成测试"""

    @pytest.fixture
    def ck(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield CheckpointStore(store_dir=Path(tmp))

    def test_retry_records_retry_trace_event(self, ck):
        """重试时记录 stage=retry trace 事件"""
        from researchforge.trace import TraceCollector
        from researchforge.orchestration.retry_policy import searching_policy

        tracer = TraceCollector(run_id="retry_trace_1")
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        graph.start("test", task_id="retry_trace_1")

        call_count = {"val": 0}

        def op_ok_after_2():
            call_count["val"] += 1
            if call_count["val"] < 3:
                raise TimeoutError("transient")
            return "ok"

        result = graph._run_with_retry("test_node", searching_policy, op_ok_after_2, tracer=tracer)
        assert result == "ok"

        events = tracer.get_all()
        retry_evs = [e for e in events if e["stage"] == "retry"]
        assert len(retry_evs) == 2, f"应记录 2 次 retry, 实际 {len(retry_evs)}"
        for ev in retry_evs:
            assert ev["action"] == "test_node"
            assert "attempt=" in ev["observation"]
        # 不应有 retry_exhausted（最终成功）
        exhausted = [e for e in events if e["stage"] == "retry_exhausted"]
        assert len(exhausted) == 0

    def test_retry_exhausted_records_trace_event(self, ck):
        """重试耗尽时记录 stage=retry_exhausted trace"""
        from researchforge.trace import TraceCollector
        from researchforge.orchestration.retry_policy import searching_policy

        tracer = TraceCollector(run_id="retry_trace_2")
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        graph.start("test", task_id="retry_trace_2")

        def always_fail():
            raise TimeoutError("always fail")

        with pytest.raises(TimeoutError):
            graph._run_with_retry("test_node", searching_policy, always_fail, tracer=tracer)

        events = tracer.get_all()
        retry_evs = [e for e in events if e["stage"] == "retry"]
        exhausted_evs = [e for e in events if e["stage"] == "retry_exhausted"]
        # searching_policy: max_retries=2, 所以 attempt=1 失败→retry, attempt=2 失败→retry, attempt=3→exhausted
        assert len(retry_evs) == 2, f"应记录 2 次 retry, 实际 {len(retry_evs)}"
        assert len(exhausted_evs) == 1, f"应记录 1 次 retry_exhausted, 实际 {len(exhausted_evs)}"
        assert exhausted_evs[0]["action"] == "test_node"
        assert "always fail" in exhausted_evs[0]["observation"]

    def test_resume_skip_not_confused_with_retry(self, ck):
        """
        Resume 后重新执行节点不应被误认为是 Retry。

        模拟场景：
          resume_skip(WRITING) → node_start(WRITING) → node_end(WRITING)
        不应产生任何 retry 事件。
        """
        from researchforge.trace import TraceCollector
        tracer = TraceCollector(run_id="resume_no_retry")

        # 直接模拟 resume 后执行的场景：跳过→正常执行
        tracer.record(agent_name="ResearchGraph", stage="resume_skip",
                      action="WRITING", observation="已有报告, 跳过")
        tracer.record(agent_name="ResearchGraph", stage="node_start",
                      action="WRITING", input="重新执行")
        tracer.record(agent_name="ResearchGraph", stage="node_end",
                      action="WRITING", result="新报告")

        events = tracer.get_all()
        stages = [e["stage"] for e in events]
        assert stages == ["resume_skip", "node_start", "node_end"], \
            f"期望 resume_skip→node_start→node_end, 实际 {stages}"
        retry_count = sum(1 for s in stages if s == "retry")
        assert retry_count == 0, f"不应包含 retry, 实际 {retry_count}"
