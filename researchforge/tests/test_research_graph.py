"""Research Graph 测试（适配新状态机）"""

import pytest
from researchforge.orchestration import State, ResearchGraph, ResearchMode


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
        def mock_write(rs, llm=None):
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
