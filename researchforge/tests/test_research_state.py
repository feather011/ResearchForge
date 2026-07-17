"""
ResearchState 序列化/反序列化测试
"""

import json
import pytest
from researchforge.orchestration.research_state import (
    ResearchState, ResearchMode, Source, Document,
    Evidence, Claim, Conflict,
)


class TestResearchStateSerialization:
    """测试 ResearchState 的 to_dict / from_dict 完整闭环"""

    def test_empty_state_roundtrip(self):
        """空状态序列化后再恢复，字段一致"""
        original = ResearchState()
        d = original.to_dict()
        restored = ResearchState.from_dict(d)

        assert restored.mode == original.mode
        assert restored.topic == original.topic
        assert restored.questions == []
        assert restored.sources == []
        assert restored.documents == []
        assert restored.evidences == []
        assert restored.claims == []
        assert restored.conflicts == []
        assert restored.report == ""
        assert restored.task_id == ""
        assert restored.current_node == ""
        assert restored.completed_nodes == []
        assert restored.failed_node == ""
        assert restored.status == "created"

    def test_full_state_roundtrip(self):
        """完整状态序列化后再恢复，所有字段一致"""
        original = ResearchState(
            mode=ResearchMode.DEEP,
            topic="RLHF技术研究",
            questions=["RLHF原理", "PPO优化", "奖励模型"],
            sources=[Source(id="s1", url="http://example.com", title="测试来源", snippet="摘要", relevance_score=0.9)],
            documents=[Document(source_id="s1", content="测试正文内容" * 10, url="http://example.com", title="文档标题")],
            evidences=[Evidence(id="ev_0", source_id="s1", text="测试证据片段", claim="结论1", relevance=0.95)],
            claims=[Claim(text="核心结论1", evidence_ids=["ev_0", "ev_1"], confidence=0.85)],
            conflicts=[Conflict(claim="冲突项", source_a="s1", source_b="s2", description="两来源数据不一致")],
            report="# 研究报告\n\n测试报告内容",
            task_id="test_001",
            current_node="WRITING",
            completed_nodes=["PLANNING", "SEARCHING", "FETCHING", "EXTRACTING"],
            status="running",
        )

        d = original.to_dict()
        restored = ResearchState.from_dict(d)

        # ▸ 标量
        assert restored.mode == original.mode
        assert restored.topic == original.topic
        assert restored.task_id == "test_001"
        assert restored.current_node == "WRITING"
        assert restored.status == "running"

        # ▸ 字符串列表
        assert restored.questions == original.questions
        assert restored.completed_nodes == ["PLANNING", "SEARCHING", "FETCHING", "EXTRACTING"]

        # ▸ Source
        assert len(restored.sources) == 1
        assert restored.sources[0].id == "s1"
        assert restored.sources[0].title == "测试来源"
        assert restored.sources[0].relevance_score == 0.9

        # ▸ Document
        assert len(restored.documents) == 1
        assert restored.documents[0].source_id == "s1"
        assert restored.documents[0].content.startswith("测试正文")

        # ▸ Evidence
        assert len(restored.evidences) == 1
        assert restored.evidences[0].id == "ev_0"
        assert restored.evidences[0].relevance == 0.95

        # ▸ Claim
        assert len(restored.claims) == 1
        assert restored.claims[0].text == "核心结论1"
        assert restored.claims[0].evidence_ids == ["ev_0", "ev_1"]
        assert restored.claims[0].confidence == 0.85

        # ▸ Conflict
        assert len(restored.conflicts) == 1
        assert restored.conflicts[0].claim == "冲突项"
        assert restored.conflicts[0].description == "两来源数据不一致"

    def test_mode_enum_preserved(self):
        """mode 枚举类型在序列化后保留为 Enum 而非字符串"""
        original = ResearchState(mode=ResearchMode.DEEP)
        d = original.to_dict()
        # to_dict 后 mode 是 str "deep"（asdict 对 str Enum 的处理）
        assert d["mode"] == "deep"

        # from_dict 后恢复为 Enum
        restored = ResearchState.from_dict(d)
        assert isinstance(restored.mode, ResearchMode)
        assert restored.mode == ResearchMode.DEEP

    def test_json_roundtrip(self):
        """to_json → from_json 完整闭环"""
        original = ResearchState(
            mode=ResearchMode.STANDARD,
            topic="测试主题",
            questions=["问题1"],
            sources=[Source(id="s1", title="测试", snippet="摘要")],
            evidences=[Evidence(id="ev_0", source_id="s1", text="证据")],
            claims=[Claim(text="结论", evidence_ids=["ev_0"])],
            report="报告内容",
            task_id="json_test",
            completed_nodes=["PLANNING"],
            status="running",
        )

        json_str = original.to_json()
        assert isinstance(json_str, str)

        # JSON 可反序列化为 dict
        parsed = json.loads(json_str)
        assert parsed["topic"] == "测试主题"
        assert parsed["mode"] == "standard"
        assert parsed["task_id"] == "json_test"
        assert len(parsed["sources"]) == 1
        assert len(parsed["evidences"]) == 1
        assert len(parsed["claims"]) == 1

        # from_json 恢复完整状态
        restored = ResearchState.from_json(json_str)
        assert restored.topic == "测试主题"
        assert isinstance(restored.mode, ResearchMode)
        assert restored.mode == ResearchMode.STANDARD
        assert restored.task_id == "json_test"
        assert len(restored.sources) == 1
        assert restored.sources[0].id == "s1"
        assert len(restored.evidences) == 1
        assert len(restored.claims) == 1
        assert restored.claims[0].text == "结论"

    def test_mark_node_lifecycle(self):
        """mark_node_start/end/failed 方法正常工作"""
        rs = ResearchState(task_id="lifecycle_test")

        rs.mark_node_start("PLANNING")
        assert rs.current_node == "PLANNING"
        assert rs.status == "running"
        assert rs.updated_at != ""

        rs.mark_node_end("PLANNING")
        assert rs.current_node == ""
        assert "PLANNING" in rs.completed_nodes

        rs.mark_node_start("SEARCHING")
        rs.mark_node_failed("SEARCHING")
        assert rs.failed_node == "SEARCHING"
        assert rs.status == "failed"

    def test_empty_fields_roundtrip(self):
        """空列表和空字符串字段在序列化后保持默认值"""
        original = ResearchState()
        d = original.to_dict()
        restored = ResearchState.from_dict(d)

        assert restored.questions == []
        assert restored.completed_nodes == []
        assert restored.sources == []
        assert restored.documents == []
        assert restored.evidences == []
        assert restored.claims == []
        assert restored.conflicts == []
        assert restored.report == ""
        assert restored.task_id == ""
        assert restored.current_node == ""
        assert restored.failed_node == ""
        assert restored.status == "created"

    def test_from_dict_missing_optional_fields(self):
        """from_dict 缺少可选字段时使用默认值"""
        # 只有必填核心字段
        minimal = {"topic": "最小测试", "mode": "fast"}
        rs = ResearchState.from_dict(minimal)

        assert rs.topic == "最小测试"
        assert rs.mode == ResearchMode.FAST
        assert rs.questions == []
        assert rs.completed_nodes == []
        assert rs.status == "created"
        assert rs.raw_searches == []
        assert rs.raw_analysis == ""

    def test_from_dict_missing_all_fields(self):
        """from_dict 空字典时全部使用默认值"""
        rs = ResearchState.from_dict({})
        # 所有字段应为默认值，不应抛异常
        assert rs.mode == ResearchMode.STANDARD
        assert rs.topic == ""
        assert rs.status == "created"
        assert rs.questions == []
        assert rs.sources == []

    def test_complex_nested_objects(self):
        """嵌套对象列表反序列化后类型正确"""
        d = {
            "topic": "嵌套测试",
            "mode": "deep",
            "sources": [
                {"id": "s1", "url": "http://a.com", "title": "A"},
                {"id": "s2", "url": "http://b.com", "title": "B", "relevance_score": 0.5},
            ],
            "claims": [
                {"text": "结论1", "evidence_ids": ["ev_0"]},
                {"text": "结论2", "evidence_ids": ["ev_1"], "confidence": 0.9},
            ],
            "conflicts": [
                {"claim": "冲突", "source_a": "s1", "source_b": "s2", "description": "描述"},
            ],
            "task_id": "nested_test",
            "completed_nodes": ["PLANNING", "SEARCHING"],
            "status": "running",
        }
        rs = ResearchState.from_dict(d)

        # 验证类型
        assert isinstance(rs.sources[0], Source)
        assert isinstance(rs.sources[1], Source)
        assert isinstance(rs.claims[0], Claim)
        assert isinstance(rs.conflicts[0], Conflict)

        # 验证值
        assert rs.sources[0].title == "A"
        assert rs.sources[1].relevance_score == 0.5
        assert rs.claims[1].confidence == 0.9
        assert len(rs.completed_nodes) == 2
        assert rs.status == "running"

    def test_old_state_compatibility(self):
        """没有新增字段的旧 dict 也能正确反序列化"""
        old_dict = {
            "mode": "standard",
            "topic": "兼容性测试",
            "questions": ["问题1"],
        }
        rs = ResearchState.from_dict(old_dict)

        # 旧字段保留
        assert rs.topic == "兼容性测试"
        # 新字段使用默认值
        assert rs.task_id == ""
        assert rs.current_node == ""
        assert rs.completed_nodes == []
        assert rs.status == "created"
