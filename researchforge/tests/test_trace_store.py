"""
TraceStore 单元测试
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from researchforge.trace import TraceEvent, TraceCollector, TraceStore


class TestTraceStore:
    """TraceStore 读写测试"""

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield TraceStore(store_dir=Path(tmp))

    def make_event(self, i=0) -> dict:
        return {
            "timestamp": 1000.0 + i,
            "run_id": "test_trace",
            "agent_name": "ResearchGraph",
            "stage": "node_start",
            "action": f"STEP_{i}",
            "thought": "",
            "tool_name": "",
            "input": "",
            "observation": "",
            "result": "",
            "duration_ms": 0.0,
        }

    def test_append_and_load(self, store: TraceStore):
        """写入和读取一条事件"""
        store.append("test_001", self.make_event(1))
        events = store.load("test_001")
        assert len(events) == 1
        assert events[0]["action"] == "STEP_1"
        assert events[0]["run_id"] == "test_trace"

    def test_multi_events_order(self, store: TraceStore):
        """多事件按写入顺序读取"""
        for i in range(5):
            store.append("order_test", self.make_event(i))
        events = store.load("order_test")
        assert len(events) == 5
        actions = [e["action"] for e in events]
        assert actions == ["STEP_0", "STEP_1", "STEP_2", "STEP_3", "STEP_4"]

    def test_nonexistent_task(self, store: TraceStore):
        """不存在的 task_id 返回空列表"""
        events = store.load("nonexistent_id")
        assert events == []

    def test_delete(self, store: TraceStore):
        """删除 Trace 文件"""
        store.append("del_test", self.make_event(0))
        assert store.count("del_test") == 1
        deleted = store.delete("del_test")
        assert deleted is True
        assert store.load("del_test") == []

    def test_delete_nonexistent(self, store: TraceStore):
        """删除不存在的文件返回 False"""
        assert store.delete("no_such_file") is False

    def test_empty_task_id_raises(self, store: TraceStore):
        """空 task_id 应当抛出 ValueError"""
        with pytest.raises(ValueError, match="不能为空"):
            store._path("")

    def test_corrupted_line_skipped(self, store: TraceStore):
        """损坏的行跳过，不影响其他事件"""
        p = store._path("corrupt_test")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"action": "OK"}\n')
            f.write("这不是 JSON\n")
            f.write('{"action": "ALSO_OK"}\n')
        events = store.load("corrupt_test")
        assert len(events) == 2, f"应跳过损坏行, 实际读取 {len(events)} 条"
        assert events[0]["action"] == "OK"
        assert events[1]["action"] == "ALSO_OK"


class TestTraceStoreMultiThread:
    """多线程写入测试"""

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield TraceStore(store_dir=Path(tmp))

    def test_concurrent_append_no_loss(self, store: TraceStore):
        """多线程并发写入不丢失事件"""
        import threading
        import time

        n_threads = 4
        events_per_thread = 25
        task_id = "concurrent_test"
        barrier = threading.Barrier(n_threads)
        errors = []

        def _writer(idx):
            barrier.wait()  # 同时开始
            for i in range(events_per_thread):
                ev = {
                    "timestamp": time.time(),
                    "run_id": task_id,
                    "agent_name": f"W{idx}",
                    "stage": "action",
                    "action": f"W{idx}_event_{i}",
                    "thought": "", "tool_name": "", "input": "",
                    "observation": "", "result": "", "duration_ms": 0.0,
                }
                try:
                    store.append(task_id, ev)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=_writer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"写入异常: {errors}"
        events = store.load(task_id)
        expected = n_threads * events_per_thread
        assert len(events) == expected, (
            f"期望 {expected} 条, 实际 {len(events)} 条"
        )


class TestCollectorWithStore:
    """TraceCollector + TraceStore 集成测试"""

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield TraceStore(store_dir=Path(tmp))

    def test_collector_persists_to_store(self, store: TraceStore):
        """TraceCollector 配置 store 后自动持久化"""
        tracer = TraceCollector(run_id="integ_test", store=store)
        tracer.record(agent_name="TestAgent", stage="think", thought="思考中")
        tracer.record(agent_name="TestAgent", stage="action", action="do_something")

        events = store.load("integ_test")
        assert len(events) == 2
        assert events[0]["stage"] == "think"
        assert events[1]["stage"] == "action"

    def test_collector_without_store(self):
        """未配置 store 时行为不变"""
        tracer = TraceCollector(run_id="no_store_test")
        tracer.record(agent_name="Test", stage="think", thought="无持久化")
        events = tracer.get_all()
        assert len(events) == 1
        assert events[0]["thought"] == "无持久化"

    def test_collector_store_and_callback(self, store: TraceStore):
        """配置 store + callback 时两者都触发"""
        cb_events = []

        def cb(ev):
            cb_events.append(ev)

        tracer = TraceCollector(run_id="dual_test", callback=cb, store=store)
        tracer.record(agent_name="Test", stage="action", action="dual")

        assert len(cb_events) == 1
        assert cb_events[0].action == "dual"
        stored = store.load("dual_test")
        assert len(stored) == 1
        assert stored[0]["action"] == "dual"

    def test_collector_store_failure_does_not_block(self, store: TraceStore):
        """存储失败不阻断主流程"""
        # 构造一个会导致写入失败的 store（只读目录）
        import os
        with tempfile.TemporaryDirectory() as tmp:
            readonly = Path(tmp) / "traces"
            readonly.mkdir()
            # 不需要真的设为只读，用无效路径模拟即可
            bad_store = TraceStore(store_dir=readonly / "nested" / "not_exist")

            tracer = TraceCollector(run_id="fail_test", store=bad_store)
            # 即使写入失败，事件仍应在内存中
            ev = tracer.record(agent_name="Test", stage="think", thought="不阻塞")
            assert ev is not None
            assert len(tracer.get_all()) == 1


class TestExistingTraceCollector:
    """确认原 TraceCollector 不受影响"""

    def test_original_behavior(self):
        """不传 store 时，get_all 和 callback 行为不变"""
        cb = Mock()
        tracer = TraceCollector(run_id="original", callback=cb)
        tracer.record(agent_name="Test", stage="action", action="original_check")
        assert len(tracer.get_all()) == 1
        assert tracer.get_all()[0]["action"] == "original_check"
        assert cb.call_count == 1


class TestTraceDuration:
    """Trace 计时测试"""

    def test_node_duration_auto_timed(self):
        """node_start → node_end 自动计算 duration_ms > 0"""
        tracer = TraceCollector(run_id="dur_test")
        tracer.record(agent_name="Graph", stage="node_start", action="PLANNING")
        import time; time.sleep(0.01)
        ev = tracer.record(agent_name="Graph", stage="node_end", action="PLANNING")
        assert ev.duration_ms > 0, f"duration_ms should be > 0, got {ev.duration_ms}"
        assert ev.duration_ms > 5, f"at least ~10ms, got {ev.duration_ms}"

    def test_failure_node_duration(self):
        """失败节点也有耗时"""
        tracer = TraceCollector(run_id="fail_dur")
        tracer.record(agent_name="Graph", stage="node_start", action="FETCHING")
        import time; time.sleep(0.01)
        # Simulate failure: mark_node_failed triggers tracer with duration_ms=0.0
        # The auto-timer should calculate the elapsed time
        ev = tracer.record(agent_name="Graph", stage="node_end", action="FETCHING", duration_ms=0.0)
        assert ev.duration_ms > 0, f"failure duration should be > 0, got {ev.duration_ms}"

    def test_retry_duration_includes_wait(self):
        """重试节点耗时包含等待时间"""
        tracer = TraceCollector(run_id="retry_dur")
        tracer.record(agent_name="Graph", stage="node_start", action="SEARCHING")
        # simulate retry: wait 0.05s, then succeed
        import time; time.sleep(0.05)
        ev = tracer.record(agent_name="Graph", stage="node_end", action="SEARCHING")
        assert ev.duration_ms > 40, f"should include retry wait, got {ev.duration_ms}"

    def test_deep_worker_independent_timing(self):
        """Deep Worker 各自独立计时"""
        tracer = TraceCollector(run_id="worker_dur")
        import time
        # Simulate two workers running in parallel with different durations
        tracer.record(agent_name="Worker1", stage="node_start", action="W1_SEARCH")
        time.sleep(0.02)
        ev1 = tracer.record(agent_name="Worker1", stage="node_end", action="W1_SEARCH")

        tracer.record(agent_name="Worker2", stage="node_start", action="W2_SEARCH")
        time.sleep(0.05)
        ev2 = tracer.record(agent_name="Worker2", stage="node_end", action="W2_SEARCH")

        assert ev1.duration_ms > 15, f"W1 should be ~20ms, got {ev1.duration_ms}"
        assert ev2.duration_ms > 40, f"W2 should be ~50ms, got {ev2.duration_ms}"
        # W2 should be longer than W1
        assert ev2.duration_ms > ev1.duration_ms, f"W2({ev2.duration_ms}) should be > W1({ev1.duration_ms})"


class TestWorkerFailureTrace:
    """Worker 失败耗时测试"""

    def test_worker_failure_records_timing(self):
        """Worker 失败时记录 node_end + duration_ms > 0 + 异常信息"""
        tracer = TraceCollector(run_id="wf_time")
        import time

        # 模拟 Worker 搜索启动 → 搜索失败（模拟抛出异常的情况）
        tracer.record(agent_name="WorkerW1", stage="node_start", action="W1_search")
        time.sleep(0.01)
        # 模拟失败：exception handler 记录 node_end，duration_ms=0.0 触发自动计时
        ev = tracer.record(agent_name="WorkerW1", stage="node_end",
                           action="W1_search",
                           observation="status=failed, error=ConnectionError('timeout')",
                           duration_ms=0.0)
        assert ev.duration_ms > 0, f"Worker 失败耗时应为正数, got {ev.duration_ms}"
        assert ev.duration_ms > 8, f"Worker 失败约 10ms, got {ev.duration_ms}"
        assert "status=failed" in ev.observation, f"缺少 status=failed, got {ev.observation}"
        assert "ConnectionError" in ev.observation, f"缺少异常信息, got {ev.observation}"

    def test_concurrent_workers_independent_failure_timing(self):
        """并发 Worker 失败计时互不覆盖"""
        tracer = TraceCollector(run_id="wf_con")
        import time

        # 模拟两个 Worker 几乎同时开始搜索
        tracer.record(agent_name="WorkerW1", stage="node_start", action="W1_search")
        tracer.record(agent_name="WorkerW2", stage="node_start", action="W2_search")
        time.sleep(0.03)
        # Worker2 先失败
        ev2 = tracer.record(agent_name="WorkerW2", stage="node_end",
                            action="W2_search",
                            observation="status=failed, error=TimeoutError()",
                            duration_ms=0.0)
        time.sleep(0.02)
        # Worker1 后失败（总 ~50ms vs ~30ms）
        ev1 = tracer.record(agent_name="WorkerW1", stage="node_end",
                            action="W1_search",
                            observation="status=failed, error=ConnectionError('refused')",
                            duration_ms=0.0)

        assert ev1.duration_ms > 40, f"W1(先start后end) 应 ~50ms, got {ev1.duration_ms}"
        assert ev2.duration_ms > 25, f"W2(后start先end) 应 ~30ms, got {ev2.duration_ms}"
        assert ev1.duration_ms > ev2.duration_ms, (
            f"W1({ev1.duration_ms}) 应 > W2({ev2.duration_ms})"
        )
