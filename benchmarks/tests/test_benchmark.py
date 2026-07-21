"""
Benchmark 框架测试
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from benchmarks.benchmark_runner import load_cases, run_single_mode, run_benchmark
from benchmarks.benchmark_report import build_report, format_report_text


class TestLoadCases:
    """case 文件读取测试"""

    def test_load_all_cases(self):
        cases = load_cases()
        assert len(cases) >= 3
        ids = [c["id"] for c in cases]
        assert "simple_research" in ids
        assert "multi_source" in ids
        assert "deep_research" in ids

    def test_load_specific_case(self):
        cases = load_cases(case_ids=["simple_research"])
        assert len(cases) == 1
        assert cases[0]["id"] == "simple_research"

    def test_load_nonexistent_case(self):
        cases = load_cases(case_ids=["no_such_case"])
        assert cases == []

    def test_case_has_required_fields(self):
        cases = load_cases()
        for c in cases:
            assert "id" in c
            assert "topic" in c
            assert "modes" in c
            assert isinstance(c["modes"], list)

    def test_empty_cases_dir(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr("benchmarks.benchmark_runner._CASES_DIR", Path(tmp))
            cases = load_cases()
            assert cases == []


class TestRunSingleMode:
    """单模式执行测试"""

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.generate.return_value = "mocked response"
        return llm

    @pytest.fixture
    def simple_case(self):
        return {"id": "test_case", "topic": "测试主题"}

    def _mock_all_nodes(self, monkeypatch):
        """Mock 全部节点使 pipeline 不依赖真实 LLM"""
        from researchforge.nodes.plan_node import run_plan_node
        from researchforge.nodes.search_node import run_search_node
        from researchforge.nodes.fetch_node import run_fetch_node
        from researchforge.nodes.extract_node import run_extract_node
        from researchforge.nodes.synthesis_node import run_synthesis_node
        from researchforge.nodes.write_node import run_write_node

        monkeypatch.setattr("researchforge.nodes.plan_node.run_plan_node",
                            lambda t, llm=None: ["问题1"])
        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node",
                            lambda q, **kw: [Mock(id="来源1")])
        monkeypatch.setattr("researchforge.nodes.fetch_node.run_fetch_node",
                            lambda src, **kw: [Mock(content="内容", source_id="s1")])
        monkeypatch.setattr("researchforge.nodes.extract_node.run_extract_node",
                            lambda docs, q: [Mock(id="ev_1", text="证据")])
        monkeypatch.setattr("researchforge.nodes.synthesis_node.run_synthesis_node",
                            lambda rs, llm=None: [Mock(text="结论", evidence_ids=["ev_1"])])
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node",
                            lambda rs, llm=None, extra_instructions="", mode="fast": "测试报告")

    def test_run_fast_mode(self, monkeypatch, mock_llm, simple_case):
        """Fast 模式执行成功"""
        self._mock_all_nodes(monkeypatch)
        result = run_single_mode(simple_case, "fast", mock_llm)
        assert result["mode"] == "fast"
        assert result["success"] is True
        assert result["duration_s"] >= 0
        assert "stats" in result

    def test_run_standard_mode(self, monkeypatch, mock_llm, simple_case):
        """Standard 模式执行成功"""
        self._mock_all_nodes(monkeypatch)
        result = run_single_mode(simple_case, "standard", mock_llm)
        assert result["mode"] == "standard"
        assert result["success"] is True

    def test_stats_contains_key_fields(self, monkeypatch, mock_llm, simple_case):
        """stats 包含关键字段"""
        self._mock_all_nodes(monkeypatch)
        result = run_single_mode(simple_case, "fast", mock_llm)
        stats = result["stats"]
        assert "sources" in stats
        assert "documents" in stats
        assert "evidences" in stats
        assert "claims" in stats
        assert "citation" in stats
        assert "execution" in stats

    def test_trace_count_recorded(self, monkeypatch, mock_llm, simple_case):
        """trace 数量记录"""
        self._mock_all_nodes(monkeypatch)
        result = run_single_mode(simple_case, "fast", mock_llm)
        assert result["trace_count"] > 0


class TestRunBenchmark:
    """多模式 benchmark 测试"""

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.generate.return_value = "mocked"
        return llm

    @pytest.fixture
    def multi_mode_case(self):
        return {"id": "test_bench", "topic": "测试", "modes": ["fast", "standard", "deep"]}

    def _mock_all(self, monkeypatch):
        from researchforge.nodes.plan_node import run_plan_node
        from researchforge.nodes.search_node import run_search_node
        from researchforge.nodes.fetch_node import run_fetch_node
        from researchforge.nodes.extract_node import run_extract_node
        from researchforge.nodes.synthesis_node import run_synthesis_node
        from researchforge.nodes.write_node import run_write_node

        monkeypatch.setattr("researchforge.nodes.plan_node.run_plan_node",
                            lambda t, llm=None: ["q1"])
        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node",
                            lambda q, **kw: [Mock(id="来源1")])
        monkeypatch.setattr("researchforge.nodes.fetch_node.run_fetch_node",
                            lambda src, **kw: [Mock(content="c", source_id="s1")])
        monkeypatch.setattr("researchforge.nodes.extract_node.run_extract_node",
                            lambda docs, q: [Mock(id="e1", text="ev")])
        monkeypatch.setattr("researchforge.nodes.synthesis_node.run_synthesis_node",
                            lambda rs, llm=None: [Mock(text="c1", evidence_ids=["e1"])])
        monkeypatch.setattr("researchforge.nodes.write_node.run_write_node",
                            lambda rs, llm=None, extra_instructions="", mode="f": "报告")

    def _mock_deep(self, monkeypatch):
        """Mock Deep 模式专用节点"""
        monkeypatch.setattr("researchforge.nodes.deep_research.LeadResearcher.make_plan",
                            lambda self, topic, **kw: ["子任务1"])
        monkeypatch.setattr("researchforge.nodes.deep_research.ResearchWorker.run",
                            lambda self: {
                                "worker_id": self.worker_id,
                                "task": self.task,
                                "sources": [Mock(id="来源1")],
                                "documents": [],
                                "evidences": [],
                                "claims": [],
                            })

    def test_three_modes_run(self, monkeypatch, mock_llm, multi_mode_case):
        """三种模式全部执行"""
        self._mock_all(monkeypatch)
        self._mock_deep(monkeypatch)
        result = run_benchmark(multi_mode_case, mock_llm)
        assert result["case"] == "test_bench"
        assert "fast" in result["results"]
        assert "standard" in result["results"]
        assert "deep" in result["results"]

    def test_specific_modes_only(self, monkeypatch, mock_llm, multi_mode_case):
        """只运行指定的模式"""
        self._mock_all(monkeypatch)
        result = run_benchmark(multi_mode_case, mock_llm, modes=["fast"])
        assert "fast" in result["results"]
        assert "standard" not in result["results"]

    def test_result_format(self, monkeypatch, mock_llm, multi_mode_case):
        """结果格式正确"""
        self._mock_all(monkeypatch)
        result = run_benchmark(multi_mode_case, mock_llm, modes=["fast"])
        fast = result["results"]["fast"]
        assert "mode" in fast
        assert "success" in fast
        assert "duration_s" in fast
        assert "stats" in fast
        assert "trace_count" in fast

    def test_deep_mode(self, monkeypatch, mock_llm, multi_mode_case):
        """Deep 模式单独执行"""
        self._mock_all(monkeypatch)
        self._mock_deep(monkeypatch)
        result = run_benchmark(multi_mode_case, mock_llm, modes=["deep"])
        assert result["results"]["deep"]["mode"] == "deep"


class TestBenchmarkReport:
    """报告生成测试"""

    @pytest.fixture
    def sample_result(self):
        return {
            "case": "test",
            "topic": "测试主题",
            "results": {
                "fast": {
                    "mode": "fast",
                    "success": True,
                    "error": None,
                    "duration_s": 10.5,
                    "trace_count": 20,
                    "stats": {
                        "sources": 3,
                        "documents": 2,
                        "evidences": 5,
                        "report_length": 500,
                        "claims": {"total": 3, "supported": 2, "supported_rate": 0.6667},
                        "citation": {"total_marks": 4, "valid_rate": 1.0, "source_utilization_rate": 0.5},
                        "coverage": {"evaluated": False, "coverage_rate": None},
                        "audit": {"passed": True, "rewritten": 0, "degraded": False},
                        "execution": {"duration_s": 10.0, "retry_count": 1, "degraded_count": 0, "slowest_node": "PLANNING"},
                    },
                },
                "standard": {
                    "mode": "standard",
                    "success": True,
                    "error": None,
                    "duration_s": 30.2,
                    "trace_count": 50,
                    "stats": {
                        "sources": 5,
                        "documents": 3,
                        "evidences": 8,
                        "report_length": 1200,
                        "claims": {"total": 4, "supported": 3, "supported_rate": 0.75},
                        "citation": {"total_marks": 10, "valid_rate": 0.9, "source_utilization_rate": 0.6},
                        "coverage": {"evaluated": True, "coverage_rate": 0.6667},
                        "audit": {"passed": True, "rewritten": 0, "degraded": False},
                        "execution": {"duration_s": 30.0, "retry_count": 0, "degraded_count": 0, "slowest_node": "WRITING"},
                    },
                },
            },
        }

    def test_report_has_comparison(self, sample_result):
        """报告包含 comparison"""
        report = build_report(sample_result)
        assert report["case"] == "test"
        assert "comparison" in report
        assert "fast" in report["comparison"]
        assert "standard" in report["comparison"]

    def test_report_key_metrics(self, sample_result):
        """报告包含关键指标"""
        report = build_report(sample_result)
        fast = report["comparison"]["fast"]
        assert fast["duration_s"] == 10.5
        assert fast["stats"]["sources"] == 3
        assert fast["stats"]["claims"]["total"] == 3
        assert fast["execution"]["retry_count"] == 1

    def test_report_text_format(self, sample_result):
        """文本报告格式化"""
        report = build_report(sample_result)
        text = format_report_text(report)
        assert "Benchmark: test" in text
        assert "FAST" in text
        assert "STANDARD" in text
        assert "10.5s" in text or "10.5" in text

    def test_report_includes_quality(self, sample_result):
        """报告包含质量评分"""
        report = build_report(sample_result)
        fast = report["comparison"]["fast"]
        assert "quality" in fast["stats"]
        assert "score" in fast["stats"]["quality"]

    def test_report_failure_handling(self):
        """失败模式的报告"""
        result = {
            "case": "fail_test",
            "topic": "",
            "results": {
                "fast": {
                    "mode": "fast",
                    "success": False,
                    "error": "LLM timeout",
                    "duration_s": 0,
                    "trace_count": 0,
                    "stats": {},
                },
            },
        }
        report = build_report(result)
        fast = report["comparison"]["fast"]
        assert fast["success"] is False
        assert fast["error"] == "LLM timeout"
