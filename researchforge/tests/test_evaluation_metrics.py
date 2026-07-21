"""
EvaluationMetrics 统一质量统计测试
"""

from unittest.mock import Mock

import pytest
from researchforge.evaluation.evaluation_metrics import build_evaluation_stats, calculate_citation_metrics
from researchforge.evaluation.execution_metrics import build_execution_metrics
from researchforge.evaluation.quality_score import build_quality_score
from researchforge.orchestration import Source
from researchforge.orchestration.research_state import ResearchState, Claim, Conflict, DeepWorkerState


def _state_with_claims(claims=None, mode="fast", metadata=None):
    state = ResearchState(mode=mode, topic="test")
    state.claims = claims or []
    if metadata:
        state.metadata = metadata
    return state


class TestBuildEvaluationStats:
    """build_evaluation_stats 纯函数测试"""

    def test_full_claim_distribution(self):
        """完整 Claim 分布"""
        claims = [
            Claim(text="c1", evidence_ids=["ev_0"], confidence=1.0),
            Claim(text="c2", evidence_ids=["ev_0"], confidence=0.5),
            Claim(text="c3", evidence_ids=["ev_0"], confidence=0.0),
        ]
        state = _state_with_claims(claims, metadata={"claim_verify_unresolved_count": 1})
        stats = build_evaluation_stats(state, [], 10.5, "fast")

        assert stats["claims"]["total"] == 3
        assert stats["claims"]["supported"] == 1
        assert stats["claims"]["partially_supported"] == 1
        assert stats["claims"]["unsupported"] == 0   # 2 个 0.0 置信度，1 个是 unverified
        assert stats["claims"]["unverified"] == 1
        assert stats["claims"]["supported_rate"] == pytest.approx(0.3333, rel=0.01)

    def test_no_claims(self):
        """无 Claim 时安全默认值"""
        state = _state_with_claims([])
        stats = build_evaluation_stats(state, [], 0.0, "fast")

        assert stats["claims"]["total"] == 0
        assert stats["claims"]["supported"] == 0
        assert stats["claims"]["partially_supported"] == 0
        assert stats["claims"]["unsupported"] == 0
        assert stats["claims"]["unverified"] == 0
        assert stats["claims"]["supported_rate"] == 0.0

    def test_no_audit(self):
        """无审计时安全默认值"""
        state = _state_with_claims()
        stats = build_evaluation_stats(state, [], 0.0, "fast")

        assert stats["audit"]["passed"] is True  # 默认
        assert stats["audit"]["rewritten"] == 0
        assert stats["audit"]["issues_count"] == 0
        assert stats["audit"]["degraded"] is False

    def test_audit_degraded(self):
        """审计降级记录"""
        state = _state_with_claims(metadata={"audit_degraded": True})
        stats = build_evaluation_stats(state, [], 0.0, "fast", audit_passed=False)

        assert stats["audit"]["passed"] is False
        assert stats["audit"]["degraded"] is True

    def test_fast_default_fields(self):
        """Fast 模式默认字段"""
        state = _state_with_claims(mode="fast")
        stats = build_evaluation_stats(state, [], 1.23, "fast")

        assert stats["sources"] == 0
        assert stats["documents"] == 0
        assert stats["evidences"] == 0
        assert stats["report_length"] == 0
        assert stats["workers"] == 0
        assert stats["conflicts"] == 0
        assert stats["execution"]["duration_s"] == 1.23
        assert stats["execution"]["retry_count"] == 0
        assert stats["execution"]["node_durations"] == {}
        assert stats["execution"]["slowest_node"] == ""

    def test_deep_workers_and_conflicts(self):
        """Deep 模式有 workers/conflicts"""
        state = ResearchState(mode="deep", topic="test")
        state.deep_workers = [
            DeepWorkerState(worker_id="W1", task="task1"),
            DeepWorkerState(worker_id="W2", task="task2"),
        ]
        state.conflicts = [Conflict(claim="c1", source_a="a", source_b="b")]

        stats = build_evaluation_stats(state, [], 0.0, "deep")

        assert stats["workers"] == 2
        assert stats["conflicts"] == 1

    def test_standard_works(self):
        """Standard 模式正常运作"""
        state = _state_with_claims(mode="standard")
        state.sources = [Mock(id="s1")]
        state.documents = [Mock()]
        state.evidences = [Mock()]
        state.report = "test report content"

        stats = build_evaluation_stats(state, [], 5.0, "standard")

        assert stats["sources"] == 1
        assert stats["documents"] == 1
        assert stats["evidences"] == 1
        assert stats["report_length"] > 0


class TestCitationMetrics:
    """calculate_citation_metrics 纯函数测试"""

    def test_all_valid_citations(self):
        """全部引用有效"""
        report = "根据[来源1]和[来源2]的数据，以及[来源1]的其他分析..."
        sources = [Source(id="来源1"), Source(id="来源2"), Source(id="来源3")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 3
        assert result["valid_marks"] == 3
        assert result["invalid_marks"] == 0
        assert result["unique_sources_cited"] == 2  # 来源1 被引2次，但只算1个 unique
        assert result["total_sources"] == 3
        assert result["valid_rate"] == 1.0

    def test_mixed_valid_invalid(self):
        """有效与无效引用混合"""
        report = "根据[来源1]和[来源9]的分析，[来源3]也支持..."
        sources = [Source(id="来源1"), Source(id="来源2"), Source(id="来源3")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 3
        assert result["valid_marks"] == 2  # 来源1, 来源3
        assert result["invalid_marks"] == 1  # 来源9
        assert result["unique_sources_cited"] == 2
        assert result["valid_rate"] == pytest.approx(0.6667, rel=0.01)

    def test_duplicate_citations_counted(self):
        """重复引用重复计数"""
        report = "如[来源1]所说，[来源1]再次强调，[来源1]总结..."
        sources = [Source(id="来源1")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 3
        assert result["valid_marks"] == 3
        assert result["unique_sources_cited"] == 1  # 只有1个 unique
        assert result["valid_rate"] == 1.0

    def test_no_citations_in_report(self):
        """报告无引用"""
        report = "这是一篇没有任何引用标记的研究报告。"
        sources = [Source(id="来源1")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 0
        assert result["valid_marks"] == 0
        assert result["invalid_marks"] == 0
        assert result["unique_sources_cited"] == 0
        assert result["valid_rate"] == 0.0

    def test_no_sources(self):
        """sources 为空"""
        report = "根据[来源1]分析..."
        result = calculate_citation_metrics(report, [])

        assert result["total_marks"] == 1
        assert result["valid_marks"] == 0
        assert result["invalid_marks"] == 1
        assert result["unique_sources_cited"] == 0
        assert result["total_sources"] == 0
        assert result["source_utilization_rate"] == 0.0

    def test_regular_brackets_not_matched(self):
        """普通中括号文本不被识别"""
        report = "在[研究背景]中，[来源1]提到了重要发现。[摘要]如下..."
        sources = [Source(id="来源1")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 1  # 只有 [来源1] 被识别
        assert result["valid_marks"] == 1
        assert result["invalid_marks"] == 0

    def test_source_utilization_rate(self):
        """source_utilization_rate 正确计算"""
        report = "根据[来源1]分析..."
        sources = [Source(id="来源1"), Source(id="来源2"), Source(id="来源3")]
        result = calculate_citation_metrics(report, sources)

        assert result["unique_sources_cited"] == 1
        assert result["total_sources"] == 3
        assert result["source_utilization_rate"] == pytest.approx(0.3333, rel=0.01)

    def test_empty_report(self):
        """空报告"""
        result = calculate_citation_metrics("", [Source(id="来源1")])
        assert result["total_marks"] == 0
        assert result["valid_rate"] == 0.0

    # ── 非标准 ID 格式 ──

    def test_source_underscore_id(self):
        """source_1 格式 ID"""
        report = "结论来自[source_1]和[来源2]的分析。"
        sources = [Source(id="source_1"), Source(id="来源2")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 2
        assert result["valid_marks"] == 2
        assert result["unique_sources_cited"] == 2

    def test_source_hyphen_id(self):
        """src-abc 格式 ID"""
        report = "如[src-abc]所述，[src-abc]是关键。"
        sources = [Source(id="src-abc"), Source(id="来源1")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 2
        assert result["valid_marks"] == 2
        assert result["unique_sources_cited"] == 1  # src-abc 引了2次

    def test_special_char_id(self):
        """特殊字符 ID（re.escape 验证）"""
        report = "见[src_a.b-c]和[来源1]。"
        sources = [Source(id="src_a.b-c"), Source(id="来源1")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 2
        assert result["valid_marks"] == 2

    def test_non_standard_id_as_invalid(self):
        """非标准 ID 不存在时标记为 invalid"""
        report = "引用了不存在的[来源9]和[unknown_id]。"
        sources = [Source(id="来源1")]
        result = calculate_citation_metrics(report, sources)

        assert result["total_marks"] == 1  # [来源9] 被标准格式匹配，[unknown_id] 不匹配任何模式
        assert result["valid_marks"] == 0
        assert result["invalid_marks"] == 1
        assert result["valid_rate"] == 0.0


class TestCoverageMetrics:
    """coverage 指标测试"""

    def _state_with_coverage_meta(self, evaluated=True, total_questions=3, gap_count=0, gap_ids=None):
        state = ResearchState(mode="standard", topic="test")
        state.claims = [Claim(text="c1")]
        state.metadata = {
            "coverage_evaluated": evaluated,
            "coverage_total_questions": total_questions,
            "coverage_gap_count": gap_count,
            "coverage_gap_ids": gap_ids or [],
        }
        return state

    def test_partial_coverage(self):
        """3 问题 2 覆盖 → coverage_rate=0.6667"""
        state = self._state_with_coverage_meta(gap_count=1, gap_ids=["问题3"])
        stats = build_evaluation_stats(state, [], 0.0, "standard")
        c = stats["coverage"]
        assert c["evaluated"] is True
        assert c["total_questions"] == 3
        assert c["covered_questions"] == 2
        assert c["gap_count"] == 1
        assert c["coverage_rate"] == pytest.approx(0.6667, rel=0.01)

    def test_full_coverage(self):
        """全部覆盖"""
        state = self._state_with_coverage_meta()
        stats = build_evaluation_stats(state, [], 0.0, "standard")
        c = stats["coverage"]
        assert c["gap_count"] == 0
        assert c["covered_questions"] == 3
        assert c["coverage_rate"] == 1.0

    def test_all_gaps(self):
        """全部缺口"""
        state = self._state_with_coverage_meta(total_questions=2, gap_count=2, gap_ids=["q1", "q2"])
        stats = build_evaluation_stats(state, [], 0.0, "standard")
        c = stats["coverage"]
        assert c["covered_questions"] == 0
        assert c["coverage_rate"] == 0.0

    def test_zero_questions_evaluated(self):
        """evaluated=True, total_questions=0 → coverage_rate=0.0"""
        state = self._state_with_coverage_meta(total_questions=0, gap_count=0)
        stats = build_evaluation_stats(state, [], 0.0, "standard")
        c = stats["coverage"]
        assert c["evaluated"] is True
        assert c["total_questions"] == 0
        assert c["covered_questions"] == 0
        assert c["gap_count"] == 0
        assert c["coverage_rate"] == 0.0

    def test_not_evaluated_rate_none(self):
        """evaluated=False → coverage_rate=None"""
        state = ResearchState(mode="fast", topic="test")
        state.claims = [Claim(text="c1")]
        state.metadata = {}
        stats = build_evaluation_stats(state, [], 0.0, "fast")
        c = stats["coverage"]
        assert c["evaluated"] is False
        assert c["coverage_rate"] is None

    def test_fast_not_evaluated(self):
        """Fast 未评估"""
        state = ResearchState(mode="fast", topic="test")
        state.claims = [Claim(text="c1")]
        state.metadata = {}
        stats = build_evaluation_stats(state, [], 0.0, "fast")
        c = stats["coverage"]
        assert c["evaluated"] is False
        assert c["total_questions"] == 0
        assert c["covered_questions"] == 0
        assert c["gap_count"] == 0
        assert c["coverage_rate"] is None


def _mock_all(monkeypatch):
    """Mock Fast 模式全部节点"""
    from researchforge.nodes.plan_node import run_plan_node
    from researchforge.nodes.search_node import run_search_node
    from researchforge.nodes.fetch_node import run_fetch_node
    from researchforge.nodes.extract_node import run_extract_node
    from researchforge.nodes.synthesis_node import run_synthesis_node
    from researchforge.nodes.write_node import run_write_node
    from researchforge.nodes.claim_verification_node import run_claim_verification_node

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


class TestExecutionMetrics:
    """build_execution_metrics 纯函数测试"""

    def test_node_durations_aggregated(self):
        """普通节点耗时统计"""
        traces = [
            {"stage": "node_start", "action": "PLANNING", "timestamp": 100.0, "duration_ms": 0.0},
            {"stage": "node_end",   "action": "PLANNING", "timestamp": 105.0, "duration_ms": 5000.0},
            {"stage": "node_start", "action": "SEARCHING", "timestamp": 105.0, "duration_ms": 0.0},
            {"stage": "node_end",   "action": "SEARCHING", "timestamp": 115.0, "duration_ms": 10000.0},
        ]
        result = build_execution_metrics(traces)
        assert result["node_durations"]["PLANNING"] == 5.0
        assert result["node_durations"]["SEARCHING"] == 10.0
        assert result["slowest_node"] == "SEARCHING"

    def test_retry_count(self):
        """retry 次数统计"""
        traces = [
            {"stage": "retry", "action": "SEARCHING"},
            {"stage": "retry", "action": "SEARCHING"},
            {"stage": "retry_exhausted", "action": "WRITING"},
        ]
        result = build_execution_metrics(traces)
        assert result["retry_count"] == 2
        assert result["retry_exhausted_count"] == 1

    def test_degraded_count(self):
        """degraded 统计"""
        traces = [
            {"stage": "degraded", "action": "CLAIM_VERIFICATION"},
            {"stage": "degraded", "action": "AUDIT"},
        ]
        result = build_execution_metrics(traces)
        assert result["degraded_count"] == 2

    def test_resume_count(self):
        """resume 统计"""
        traces = [
            {"stage": "resume_started", "action": ""},
        ]
        result = build_execution_metrics(traces)
        assert result["resume_count"] == 1

    def test_slowest_node_calculation(self):
        """多节点 slowest 计算"""
        traces = [
            {"stage": "node_start", "action": "A", "timestamp": 0.0, "duration_ms": 0.0},
            {"stage": "node_end",   "action": "A", "timestamp": 1.0, "duration_ms": 1000.0},
            {"stage": "node_start", "action": "B", "timestamp": 1.0, "duration_ms": 0.0},
            {"stage": "node_end",   "action": "B", "timestamp": 5.0, "duration_ms": 4000.0},
            {"stage": "node_start", "action": "C", "timestamp": 5.0, "duration_ms": 0.0},
            {"stage": "node_end",   "action": "C", "timestamp": 6.0, "duration_ms": 500.0},
        ]
        result = build_execution_metrics(traces)
        assert result["slowest_node"] == "B"
        assert result["node_durations"]["B"] == 4.0

    def test_deep_worker_durations(self):
        """Deep Worker 耗时"""
        traces = [
            {"stage": "node_start", "action": "W1_search", "timestamp": 0.0, "duration_ms": 0.0},
            {"stage": "node_end",   "action": "W1_search", "timestamp": 3.0, "duration_ms": 3000.0},
            {"stage": "node_start", "action": "W2_search", "timestamp": 0.0, "duration_ms": 0.0},
            {"stage": "node_end",   "action": "W2_search", "timestamp": 5.0, "duration_ms": 5000.0},
        ]
        result = build_execution_metrics(traces)
        assert result["node_durations"]["W1_search"] == 3.0
        assert result["node_durations"]["W2_search"] == 5.0
        assert result["slowest_node"] == "W2_search"

    def test_empty_traces_safe_default(self):
        """无 trace 安全默认"""
        result = build_execution_metrics([])
        assert result["duration_s"] == 0.0
        assert result["retry_count"] == 0
        assert result["retry_exhausted_count"] == 0
        assert result["degraded_count"] == 0
        assert result["resume_count"] == 0
        assert result["node_durations"] == {}
        assert result["slowest_node"] == ""

    def test_duration_s_from_traces(self):
        """trace timestamp 计算总耗时"""
        traces = [
            {"stage": "node_start", "action": "A", "timestamp": 100.0},
            {"stage": "node_end",   "action": "A", "timestamp": 150.0},
        ]
        result = build_execution_metrics(traces)
        assert result["duration_s"] == 50.0

    def test_duration_s_override(self):
        """传入 duration_s 可覆盖 trace 计算值"""
        traces = [
            {"stage": "node_start", "action": "A", "timestamp": 0.0},
            {"stage": "node_end",   "action": "A", "timestamp": 10.0},
        ]
        result = build_execution_metrics(traces, duration_s=99.5)
        assert result["duration_s"] == 99.5

    def test_integrated_in_build_evaluation_stats(self):
        """execution 集成到 build_evaluation_stats"""
        traces = [
            {"stage": "retry", "action": "SEARCHING"},
            {"stage": "retry_exhausted", "action": "WRITING"},
            {"stage": "degraded", "action": "AUDIT"},
            {"stage": "resume_started", "action": ""},
            {"stage": "node_start", "action": "PLANNING", "timestamp": 0.0, "duration_ms": 0.0},
            {"stage": "node_end",   "action": "PLANNING", "timestamp": 5.0, "duration_ms": 5000.0},
        ]
        state = ResearchState(mode="fast", topic="test")
        state.claims = [Claim(text="c1")]
        stats = build_evaluation_stats(state, traces, 10.0, "fast")
        e = stats["execution"]
        assert e["retry_count"] == 1
        assert e["retry_exhausted_count"] == 1
        assert e["degraded_count"] == 1
        assert e["resume_count"] == 1
        assert e["duration_s"] == 10.0  # 使用传入值
        assert "PLANNING" in e["node_durations"]
        assert e["slowest_node"] == "PLANNING"


class TestQualityScore:
    """build_quality_score 纯函数测试"""

    def _make_stats(self, override=None):
        """创建标准 stats 用于测试"""
        stats = {
            "claims": {"supported_rate": 0.9},
            "citation": {"valid_rate": 1.0, "source_utilization_rate": 0.8},
            "coverage": {"evaluated": True, "coverage_rate": 0.8},
            "audit": {"passed": True, "degraded": False, "issues_count": 0},
        }
        if override:
            stats.update(override)
        return stats

    def test_high_quality(self):
        """高质量报告 → A 级"""
        s = self._make_stats()
        q = build_quality_score(s)
        assert q["quality_score"] >= 90
        assert q["grade"] == "A"
        assert q["breakdown"]["claim_score"] == 100

    def test_low_quality(self):
        """低质量报告 → D/F"""
        s = self._make_stats({
            "claims": {"supported_rate": 0.3},
            "citation": {"valid_rate": 0.2, "source_utilization_rate": 0.1},
            "coverage": {"evaluated": True, "coverage_rate": 0.2},
            "audit": {"passed": False, "degraded": True, "issues_count": 5},
        })
        q = build_quality_score(s)
        assert q["quality_score"] < 60
        assert q["grade"] in ("D", "F")

    def test_no_coverage_evaluated(self):
        """未评估 Coverage 时给 50 分"""
        s = self._make_stats({"coverage": {"evaluated": False}})
        q = build_quality_score(s)
        assert q["breakdown"]["coverage_score"] == 50

    def test_audit_degraded(self):
        """Audit degraded → base 50"""
        s = self._make_stats({"audit": {"passed": False, "degraded": True, "issues_count": 0}})
        q = build_quality_score(s)
        assert q["breakdown"]["audit_score"] == 50

    def test_audit_issues_penalty(self):
        """Audit issues 每个扣 10"""
        s = self._make_stats({"audit": {"passed": False, "degraded": False, "issues_count": 3}})
        q = build_quality_score(s)
        # passed: False → base 50, 3 issues → -30 → 20
        assert q["breakdown"]["audit_score"] == 20

    def test_empty_stats(self):
        """空 stats → 全 0, F"""
        q = build_quality_score({})
        assert q["quality_score"] == 0.0
        assert q["grade"] == "F"
        assert q["breakdown"]["claim_score"] == 0

    def test_none_stats(self):
        """None stats → 全 0, F"""
        q = build_quality_score(None)
        assert q["quality_score"] == 0.0

    def test_integrated_in_stats(self):
        """quality 集成到 build_evaluation_stats 中"""
        state = ResearchState(mode="fast", topic="test")
        state.claims = [Claim(text="c1", evidence_ids=["ev_0"], confidence=1.0),
                        Claim(text="c2", evidence_ids=["ev_0"], confidence=0.5)]
        stats = build_evaluation_stats(state, [], 1.0, "fast")
        assert "quality" in stats
        assert "quality_score" in stats["quality"]
        assert "grade" in stats["quality"]
        assert "breakdown" in stats["quality"]


class TestBackwardCompatibility:
    """旧顶层字段仍存在"""

    @pytest.fixture
    def ck(self):
        import tempfile, pathlib
        from researchforge.orchestration.checkpoint_store import CheckpointStore
        with tempfile.TemporaryDirectory() as tmp:
            yield CheckpointStore(store_dir=pathlib.Path(tmp))

    def _execute_fast(self, monkeypatch, ck):
        from researchforge.orchestration import ResearchGraph, ResearchMode
        _mock_all(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.FAST, checkpoint_store=ck)
        return graph.execute("测试", llm=None, progress_callback=None)

    def test_old_claim_verification_field(self, monkeypatch, ck):
        """返回结果保留 claim_verification"""
        result = self._execute_fast(monkeypatch, ck)
        assert "claim_verification" in result
        assert result["claim_verification"]["total"] >= 0

    def test_old_audit_field(self, monkeypatch, ck):
        """返回结果保留 audit"""
        result = self._execute_fast(monkeypatch, ck)
        assert "audit" in result
        assert "passed" in result["audit"]

    def test_old_duration_s_field(self, monkeypatch, ck):
        """返回结果保留 _duration_s"""
        result = self._execute_fast(monkeypatch, ck)
        assert "_duration_s" in result
        assert result["_duration_s"] >= 0

    def test_fast_includes_citation_in_stats(self, monkeypatch, ck):
        """Fast 模式 stats 包含 citation"""
        result = self._execute_fast(monkeypatch, ck)
        assert "citation" in result["stats"]
        assert "valid_rate" in result["stats"]["citation"]

    def test_deep_includes_citation(self, monkeypatch, ck):
        """Deep 模式 stats 包含 citation"""
        from researchforge.orchestration import ResearchGraph, ResearchMode

        def mock_deep_plan(topic, **kw):
            return ["子任务1", "子任务2"]

        def mock_deep_run(worker_id, task, llm, **kw):
            from researchforge.orchestration import Source
            return {
                "worker_id": worker_id,
                "task": task,
                "sources": [Source(id="来源1")],
                "documents": [],
                "evidences": [],
                "claims": [],
            }

        monkeypatch.setattr("researchforge.nodes.deep_research.LeadResearcher.make_plan", mock_deep_plan)
        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_deep_run)
        _mock_all(monkeypatch)

        graph = ResearchGraph(mode=ResearchMode.DEEP, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)
        assert "citation" in result["stats"]
        assert result["stats"]["citation"]["total_marks"] >= 0

    def test_standard_includes_citation(self, monkeypatch, ck):
        """Standard 模式 stats 包含 citation"""
        from researchforge.orchestration import ResearchGraph, ResearchMode

        def mock_standard_write(rs, llm=None, extra_instructions="", mode="standard"):
            return "包含[来源1]和[来源2]引用的标准报告。"

        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node", mock_standard_write)
        _mock_all(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)
        assert "citation" in result["stats"]
        assert result["stats"]["citation"]["total_marks"] >= 0

    def test_standard_coverage_evaluated(self, monkeypatch, ck):
        """Standard 模式执行了 Coverage → evaluated=True"""
        from researchforge.orchestration import ResearchGraph, ResearchMode

        def mock_cov(rs):
            return True, []

        monkeypatch.setattr("researchforge.nodes.coverage_node.run_coverage_node", mock_cov)
        _mock_all(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.STANDARD, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)
        assert result["stats"]["coverage"]["evaluated"] is True
        assert result["stats"]["coverage"]["total_questions"] >= 0

    def test_deep_coverage_evaluated(self, monkeypatch, ck):
        """Deep 模式也应执行 Coverage"""
        from researchforge.orchestration import ResearchGraph, ResearchMode

        def mock_deep_plan(topic, **kw):
            return ["子任务1"]

        def mock_deep_run(worker_id, task, llm, **kw):
            return {"worker_id": worker_id, "task": task, "sources": [], "documents": [], "evidences": [], "claims": []}

        monkeypatch.setattr("researchforge.nodes.deep_research.LeadResearcher.make_plan", mock_deep_plan)
        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run", mock_deep_run)
        _mock_all(monkeypatch)
        graph = ResearchGraph(mode=ResearchMode.DEEP, checkpoint_store=ck)
        result = graph.execute("测试", llm=None, progress_callback=None)
        assert result["stats"]["coverage"]["evaluated"] is True
