"""CheckpointStore 测试"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from researchforge.orchestration.checkpoint_store import CheckpointStore
from researchforge.orchestration.research_state import (
    ResearchState, ResearchMode,
    Source, Document, Evidence, Claim, Conflict,
)


class TestCheckpointStore:
    """CheckpointStore 单元测试"""

    @pytest.fixture
    def tmp_store(self):
        """每个测试使用独立的临时目录"""
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(store_dir=Path(tmp))
            yield store
            # TemporaryDirectory 自动清理

    def make_state(self, task_id: str, mode: str = "standard") -> ResearchState:
        """创建带数据的 ResearchState"""
        state = ResearchState(
            mode=ResearchMode(mode),
            topic=f"测试主题_{task_id}",
            task_id=task_id,
        )
        state.questions = ["问题1", "问题2"]
        state.sources = [
            Source(id="s1", title="来源1", snippet="这是来源1的摘要"),
            Source(id="s2", title="来源2", snippet="这是来源2的摘要"),
        ]
        state.evidences = [
            Evidence(id="ev_0", source_id="s1", text="证据文本0"),
            Evidence(id="ev_1", source_id="s2", text="证据文本1"),
        ]
        state.claims = [
            Claim(text="核心结论1", evidence_ids=["ev_0"], confidence=1.0),
            Claim(text="核心结论2", evidence_ids=["ev_1"], confidence=0.5),
        ]
        state.mark_node_start("planning")
        return state

    # ─── 基本操作 ───

    def test_save_and_load(self, tmp_store):
        """保存后能完整加载"""
        state = self.make_state("test_save_load")
        tid = tmp_store.save(state)
        assert tid == "test_save_load"

        loaded = tmp_store.load("test_save_load")
        assert loaded is not None
        assert loaded.task_id == "test_save_load"
        assert loaded.topic == "测试主题_test_save_load"
        assert loaded.mode == ResearchMode.STANDARD
        assert len(loaded.questions) == 2
        assert len(loaded.sources) == 2
        assert len(loaded.evidences) == 2
        assert len(loaded.claims) == 2
        assert loaded.status == "running"
        assert loaded.current_node == "planning"

    def test_save_overwrite(self, tmp_store):
        """覆盖保存能正确替换"""
        s1 = self.make_state("overwrite", mode="fast")
        tmp_store.save(s1)

        # 修改后重新保存
        s2 = tmp_store.load("overwrite")
        s2.topic = "覆盖后主题"
        s2.mode = ResearchMode.STANDARD
        tmp_store.save(s2)

        loaded = tmp_store.load("overwrite")
        assert loaded.topic == "覆盖后主题"
        assert loaded.mode == ResearchMode.STANDARD

    def test_exists(self, tmp_store):
        """exists 方法正确"""
        assert not tmp_store.exists("nonexistent")
        state = self.make_state("check_exists")
        tmp_store.save(state)
        assert tmp_store.exists("check_exists")

    def test_delete(self, tmp_store):
        """删除后文件不存在"""
        state = self.make_state("to_delete")
        tmp_store.save(state)
        assert tmp_store.exists("to_delete")

        deleted = tmp_store.delete("to_delete")
        assert deleted is True
        assert not tmp_store.exists("to_delete")

        # 重复删除
        deleted2 = tmp_store.delete("to_delete")
        assert deleted2 is False

    def test_load_nonexistent(self, tmp_store):
        """不存在的 task_id 返回 None"""
        loaded = tmp_store.load("never_saved")
        assert loaded is None

    # ─── 边界与异常 ───

    def test_empty_task_id_raises(self, tmp_store):
        """task_id 为空时 save 抛出异常"""
        state = self.make_state("")
        with pytest.raises(ValueError, match="task_id 为空"):
            tmp_store.save(state)

    def test_corrupted_file(self, tmp_store):
        """损坏的文件返回 None"""
        state = self.make_state("corrupted")
        tmp_store.save(state)

        # 手动写无效 JSON
        fpath = tmp_store._path("corrupted")
        fpath.write_text("{invalid json!!!}", encoding="utf-8")

        loaded = tmp_store.load("corrupted")
        assert loaded is None

    def test_corrupted_partial_json(self, tmp_store):
        """部分 JSON 文件（写入中断）返回 None"""
        state = self.make_state("partial")
        tmp_store.save(state)

        fpath = tmp_store._path("partial")
        fpath.write_text('{"mode": "fast",', encoding="utf-8")

        loaded = tmp_store.load("partial")
        assert loaded is None

    def test_load_empty_file(self, tmp_store):
        """空文件返回 None"""
        state = self.make_state("empty_file")
        tmp_store.save(state)

        fpath = tmp_store._path("empty_file")
        fpath.write_text("", encoding="utf-8")

        loaded = tmp_store.load("empty_file")
        assert loaded is None

    def test_corrupted_not_dict(self, tmp_store):
        """文件内容不是 dict（如纯字符串）时返回 None"""
        state = self.make_state("not_dict")
        tmp_store.save(state)

        fpath = tmp_store._path("not_dict")
        fpath.write_text('"just a string"', encoding="utf-8")

        loaded = tmp_store.load("not_dict")
        assert loaded is None

    # ─── 自动化创建目录 ───

    def test_auto_create_dir(self):
        """目录不存在时自动创建"""
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / "nested" / "checkpoints"
            assert not store_dir.exists()

            store = CheckpointStore(store_dir=store_dir)
            state = self.make_state("auto_create")
            store.save(state)

            assert store_dir.exists()
            assert store.exists("auto_create")

    # ─── 路径安全 ───

    def test_path_traversal_prevention(self, tmp_store):
        """路径穿越被字符替换阻止"""
        state = self.make_state("../evil")
        tmp_store.save(state)

        # / 被替换为 _，所以文件保存在目录内而不是父目录
        expected = tmp_store._dir / ".._evil.json"
        assert expected.exists()
        assert not (tmp_store._dir.parent / "evil.json").exists()

        # 反斜杠也一样
        state2 = self.make_state("..\\evil2")
        tmp_store.save(state2)
        expected2 = tmp_store._dir / ".._evil2.json"
        assert expected2.exists()

    def test_special_chars(self, tmp_store):
        """特殊字符被安全处理"""
        state = self.make_state("test/1\\2..3")
        tmp_store.save(state)

        # 路径穿越被过滤，/_ 被替换为 _
        expected = tmp_store._dir / "test_1_2..3.json"
        assert expected.exists()
        loaded = tmp_store.load("test/1\\2..3")
        assert loaded is not None
        assert loaded.task_id == "test/1\\2..3"

    # ─── 列表与管理 ───

    def test_list_ids(self, tmp_store):
        """list_ids 返回所有 task_id"""
        for i in range(3):
            s = self.make_state(f"list_{i}")
            tmp_store.save(s)

        ids = tmp_store.list_ids()
        assert len(ids) == 3
        assert ids == ["list_0", "list_1", "list_2"]

    def test_count(self, tmp_store):
        """count 返回数量"""
        assert tmp_store.count() == 0
        for i in range(3):
            s = self.make_state(f"count_{i}")
            tmp_store.save(s)
        assert tmp_store.count() == 3

    def test_clear(self, tmp_store):
        """clear 清空所有检查点"""
        for i in range(3):
            s = self.make_state(f"clear_{i}")
            tmp_store.save(s)
        assert tmp_store.count() > 0
        tmp_store.clear()
        assert tmp_store.count() == 0

    # ─── 数据完整性 ───

    def test_full_roundtrip(self, tmp_store):
        """完整往返：复杂状态保存后与原数据一致"""
        state = self.make_state("roundtrip", mode="deep")
        state.questions = ["Q1", "Q2", "Q3"]
        state.documents = [
            Document(source_id="s1", content="文档内容1" * 10, url="http://example.com/1"),
        ]
        state.conflicts = [
            Conflict(claim="冲突结论", source_a="s1", source_b="s2", description="数据不一致"),
        ]
        state.report = "这是最终报告。" * 20
        state.mark_node_start("writing")
        state.mark_node_end("planning")
        state.mark_node_end("searching")

        tmp_store.save(state)
        loaded = tmp_store.load("roundtrip")

        assert loaded.topic == state.topic
        assert loaded.mode == state.mode
        assert loaded.questions == state.questions
        assert loaded.sources == state.sources
        assert loaded.documents == state.documents
        assert loaded.evidences == state.evidences
        assert loaded.claims == state.claims
        assert loaded.conflicts == state.conflicts
        assert loaded.report == state.report
        assert loaded.status == "running"
        assert loaded.current_node == "writing"
        assert set(loaded.completed_nodes) == {"planning", "searching"}
