"""
Demo 测试 — 验证 Mock 运行、故障注入和输出结构
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))


class TestDemoRunner:
    """Demo 运行器测试"""

    def test_mock_fast_demo_runs(self, monkeypatch):
        """Mock Fast Demo 完整运行"""
        from demo.mock_provider import setup_mock_patches
        from demo.scripts.run_demo import run_mock_demo

        setup_mock_patches(monkeypatch)
        case = {"id": "fast_simple", "topic": "Python 装饰器原理与应用", "modes": ["fast"]}
        output = run_mock_demo(case)
        assert output["status"] == "completed"
        assert output["case"] == "fast_simple"
        assert output["mode"] == "fast"
        assert output["duration_s"] > 0

    def test_mock_standard_demo_runs(self, monkeypatch):
        """Mock Standard Demo 完整运行"""
        from demo.mock_provider import setup_mock_patches
        from demo.scripts.run_demo import run_mock_demo

        setup_mock_patches(monkeypatch)
        case = {"id": "standard_multi_source", "topic": "测试", "modes": ["standard"]}
        output = run_mock_demo(case)
        assert output["status"] == "completed"
        assert output["mode"] == "standard"

    def test_mock_deep_demo_runs(self, monkeypatch):
        """Mock Deep Demo 完整运行"""
        from demo.mock_provider import setup_mock_patches
        from demo.scripts.run_demo import run_mock_demo

        setup_mock_patches(monkeypatch)
        case = {"id": "deep_complex", "topic": "测试", "modes": ["deep"]}
        output = run_mock_demo(case)
        assert output["status"] == "completed"
        assert output["mode"] == "deep"

    def test_output_has_stats(self, monkeypatch):
        """输出包含 stats"""
        from demo.mock_provider import setup_mock_patches
        from demo.scripts.run_demo import run_mock_demo

        setup_mock_patches(monkeypatch)
        case = {"id": "fast_simple", "topic": "测试", "modes": ["fast"]}
        output = run_mock_demo(case)
        stats = output.get("result", {}).get("stats", {})
        assert "claims" in stats
        assert "citation" in stats
        assert "execution" in stats
        assert "quality" in stats

    def test_output_has_traces(self, monkeypatch):
        """输出包含 traces"""
        from demo.mock_provider import setup_mock_patches
        from demo.scripts.run_demo import run_mock_demo

        setup_mock_patches(monkeypatch)
        case = {"id": "fast_simple", "topic": "测试", "modes": ["fast"]}
        output = run_mock_demo(case)
        assert len(output.get("traces", [])) > 0

    def test_output_has_report(self, monkeypatch):
        """输出包含报告"""
        from demo.mock_provider import setup_mock_patches
        from demo.scripts.run_demo import run_mock_demo

        setup_mock_patches(monkeypatch)
        case = {"id": "fast_simple", "topic": "测试", "modes": ["fast"]}
        output = run_mock_demo(case)
        assert len(output.get("result", {}).get("report", "")) > 0


class TestFaultInjection:
    """故障注入测试"""

    def test_fault_injector_default_off(self):
        """故障注入默认关闭"""
        from demo.scripts.fault_injector import get_fault_injector
        inj = get_fault_injector()
        inj.reset()
        assert inj.is_active is False

    def test_fault_injector_activate(self):
        """激活后生效"""
        from demo.scripts.fault_injector import get_fault_injector
        inj = get_fault_injector()
        inj.reset()
        inj.activate()
        assert inj.is_active is True

    def test_fault_injector_rule(self):
        """规则匹配"""
        from demo.scripts.fault_injector import get_fault_injector
        inj = get_fault_injector()
        inj.reset()
        inj.activate()
        inj.add_rule("searching", fail_count=1)
        assert inj.should_fail("searching") is True
        assert inj.should_fail("searching") is False  # 第二次不失败

    def test_fault_injector_no_rule_match(self):
        """不匹配的规则不失败"""
        from demo.scripts.fault_injector import get_fault_injector
        inj = get_fault_injector()
        inj.reset()
        inj.activate()
        inj.add_rule("searching", fail_count=1)
        assert inj.should_fail("writing") is False  # 无 writing 规则

    def test_fault_injector_triggers_retry(self, monkeypatch):
        """故障注入触发重试"""
        from demo.mock_provider import setup_mock_patches
        from researchforge.orchestration import ResearchGraph, ResearchMode
        from researchforge.trace import TraceCollector

        # 先注入所有 mock，再覆盖 search_node
        setup_mock_patches(monkeypatch)

        call_count = {"n": 0}

        def _fail_once(q, **kw):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise TimeoutError("[FaultInjector] 模拟 searching 故障")
            from researchforge.orchestration import Source
            return [Source(id="s1", title="M", snippet="m", url="")]

        monkeypatch.setattr("researchforge.nodes.search_node.run_search_node", _fail_once)

        tracer = TraceCollector(run_id="fault_test")
        graph = ResearchGraph(mode=ResearchMode.FAST)
        result = graph.execute("测试", llm=None, progress_callback=None, tracer=tracer)

        assert result["report"] is not None
        traces = tracer.get_all()
        retry_events = [t for t in traces if t.get("stage") == "retry"]
        assert len(retry_events) >= 1, f"预期至少 1 个 retry 事件, 实际 {len(retry_events)}"


class TestDemoCases:
    """Demo case 文件测试"""

    def test_cases_file_exists(self):
        cases_path = Path(__file__).resolve().parent.parent / "demo_cases.json"
        assert cases_path.exists()

    def test_cases_have_required_fields(self):
        from demo.scripts.run_demo import load_demo_cases
        cases = load_demo_cases()
        assert len(cases) >= 3
        for c in cases:
            assert "id" in c
            assert "topic" in c
            assert "modes" in c

    def test_case_modes_are_valid(self):
        from demo.scripts.run_demo import load_demo_cases
        cases = load_demo_cases()
        valid_modes = {"fast", "standard", "deep"}
        for c in cases:
            for m in c["modes"]:
                assert m in valid_modes, f"Invalid mode {m} in case {c['id']}"


class TestMockProvider:
    """Mock Provider 测试"""

    def test_create_mock_llm(self):
        from demo.mock_provider import create_mock_llm
        llm = create_mock_llm()
        result = llm.generate("测试 prompt")
        assert len(result) > 0
        assert isinstance(result, str)

    def test_setup_mock_patches(self, monkeypatch):
        from demo.mock_provider import setup_mock_patches
        result = setup_mock_patches(monkeypatch)
        assert result is True
