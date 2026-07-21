"""
Trace 查询 API 测试（独立 FastAPI 实例，不依赖 app.py 模块级代码）
"""
import json
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from researchforge.trace import TraceStore


class TestTraceAPI:
    """GET /api/research/{task_id}/traces"""

    @pytest.fixture
    def client_and_store(self):
        """创建独立 FastAPI 实例 + 临时 TraceStore"""
        tmp = tempfile.mkdtemp()
        store = TraceStore(store_dir=Path(tmp))

        test_app = FastAPI()

        @test_app.get("/api/research/{research_id}/traces")
        async def get_traces(research_id: str, stage: str = None, agent_name: str = None):
            traces = store.load(research_id)
            if stage:
                traces = [t for t in traces if t.get("stage") == stage]
            if agent_name:
                traces = [t for t in traces if t.get("agent_name") == agent_name]
            traces.sort(key=lambda t: t.get("timestamp", 0))
            return {"research_id": research_id, "traces": traces}

        client = TestClient(test_app)
        yield client, store
        shutil.rmtree(tmp, ignore_errors=True)

    def make_event(self, timestamp: float, stage: str = "node_start",
                   agent_name: str = "Graph", action: str = "PLANNING") -> dict:
        return {
            "timestamp": timestamp,
            "run_id": "test_api",
            "agent_name": agent_name,
            "stage": stage,
            "action": action,
            "thought": "",
            "tool_name": "",
            "input": "",
            "observation": "",
            "result": "",
            "duration_ms": 0.0,
        }

    # ==================== 基本功能 ====================

    def test_read_all_traces(self, client_and_store):
        """读取全部 Trace"""
        client, store = client_and_store
        store.append("task_001", self.make_event(1000.0, "node_start"))
        store.append("task_001", self.make_event(1001.0, "node_end"))

        resp = client.get("/api/research/task_001/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["research_id"] == "task_001"
        assert len(data["traces"]) == 2

    # ==================== 排序 ====================

    def test_traces_sorted_by_timestamp(self, client_and_store):
        """按 timestamp 升序返回"""
        client, store = client_and_store
        store.append("sort_test", self.make_event(3000.0, "node_end"))
        store.append("sort_test", self.make_event(1000.0, "node_start"))
        store.append("sort_test", self.make_event(2000.0, "node_start"))

        resp = client.get("/api/research/sort_test/traces")
        data = resp.json()
        timestamps = [t["timestamp"] for t in data["traces"]]
        assert timestamps == sorted(timestamps), f"期望升序, 实际 {timestamps}"
        assert timestamps == [1000.0, 2000.0, 3000.0]

    # ==================== 筛选 ====================

    def test_filter_by_stage(self, client_and_store):
        """stage 参数过滤"""
        client, store = client_and_store
        store.append("filt_test", self.make_event(1000.0, "node_start"))
        store.append("filt_test", self.make_event(1001.0, "node_end"))
        store.append("filt_test", self.make_event(1002.0, "think"))

        resp = client.get("/api/research/filt_test/traces?stage=node_end")
        data = resp.json()
        assert len(data["traces"]) == 1
        assert data["traces"][0]["stage"] == "node_end"

    def test_filter_by_agent_name(self, client_and_store):
        """agent_name 参数过滤"""
        client, store = client_and_store
        store.append("filt_test", self.make_event(1000.0, agent_name="Graph"))
        store.append("filt_test", self.make_event(1001.0, agent_name="WorkerW1"))
        store.append("filt_test", self.make_event(1002.0, agent_name="Graph"))

        resp = client.get("/api/research/filt_test/traces?agent_name=WorkerW1")
        data = resp.json()
        assert len(data["traces"]) == 1
        assert data["traces"][0]["agent_name"] == "WorkerW1"

    def test_filter_combined(self, client_and_store):
        """stage + agent_name 组合过滤"""
        client, store = client_and_store
        store.append("filt_test", self.make_event(1000.0, "node_start", "Graph"))
        store.append("filt_test", self.make_event(1001.0, "node_end", "Graph"))
        store.append("filt_test", self.make_event(1002.0, "node_end", "WorkerW1"))

        resp = client.get(
            "/api/research/filt_test/traces?stage=node_end&agent_name=Graph"
        )
        data = resp.json()
        assert len(data["traces"]) == 1
        assert data["traces"][0]["stage"] == "node_end"
        assert data["traces"][0]["agent_name"] == "Graph"

    # ==================== 不存在任务 / 空结果 ====================

    def test_nonexistent_task_returns_empty(self, client_and_store):
        """不存在的 task_id 返回空 traces 列表（非 404）"""
        client, _ = client_and_store
        resp = client.get("/api/research/no_such_task/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["research_id"] == "no_such_task"
        assert data["traces"] == []

    # ==================== 损坏数据 ====================

    def test_corrupted_lines_skipped(self, client_and_store):
        """损坏行被跳过，不影响返回"""
        client, store = client_and_store
        p = store._path("corrupt_api")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"action": "OK"}\n')
            f.write("这不是 JSON\n")
            f.write('{"action": "ALSO_OK"}\n')

        resp = client.get("/api/research/corrupt_api/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["traces"]) == 2
        assert data["traces"][0]["action"] == "OK"
        assert data["traces"][1]["action"] == "ALSO_OK"

    # ==================== 不影响主流程 ====================

    def test_readonly_does_not_write(self, client_and_store):
        """GET 查询不应写入任何内容"""
        client, store = client_and_store
        before = store.count("readonly_test")
        client.get("/api/research/readonly_test/traces")
        after = store.count("readonly_test")
        assert after == before, "查询不应产生写入"
