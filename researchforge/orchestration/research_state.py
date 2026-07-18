"""
ResearchState — 研究任务的数据中心

替代旧的 HybridMemory 作为单次研究的数据存储。
支持序列化/反序列化，为检查点做准备。
"""

import json
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum


class ResearchMode(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class Source:
    """网页来源"""
    id: str
    url: str = ""
    title: str = ""
    snippet: str = ""
    relevance_score: float = 0.0


@dataclass
class Document:
    """抓取并清洗后的网页正文"""
    source_id: str
    content: str
    url: str = ""
    title: str = ""


@dataclass
class Evidence:
    """支持结论的原文片段"""
    id: str
    source_id: str
    text: str
    claim: str = ""
    relevance: float = 0.0


@dataclass
class Claim:
    """核心结论"""
    text: str
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Conflict:
    """来源冲突"""
    claim: str
    source_a: str
    source_b: str
    description: str = ""


@dataclass
class DeepWorkerState:
    """Deep 模式 Worker 执行状态（线程安全，不可变写入）"""
    worker_id: str
    task: str
    status: str = "pending"  # pending | running | completed | failed
    sources: List[Source] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    evidences: List[Evidence] = field(default_factory=list)
    error: str = ""


def _reconstruct_deep_worker(dw: dict) -> DeepWorkerState:
    """从 dict 重建 DeepWorkerState（含嵌套 Source/Document/Evidence）"""
    return DeepWorkerState(
        worker_id=dw.get("worker_id", ""),
        task=dw.get("task", ""),
        status=dw.get("status", "pending"),
        sources=[Source(**s) for s in dw.get("sources", []) if isinstance(s, dict)],
        documents=[Document(**d) for d in dw.get("documents", []) if isinstance(d, dict)],
        evidences=[Evidence(**e) for e in dw.get("evidences", []) if isinstance(e, dict)],
        error=dw.get("error", ""),
    )


# ─── 嵌套 dataclass 类型映射（用于反序列化） ───
_NESTED_TYPES = {
    "sources": Source,
    "documents": Document,
    "evidences": Evidence,
    "claims": Claim,
    "conflicts": Conflict,
}


def _reconstruct_list(items: list, cls: type) -> list:
    """从 dict 列表重建 dataclass 列表"""
    return [cls(**item) if isinstance(item, dict) else item for item in items]


def _to_iso(dt: Optional[datetime.datetime]) -> str:
    """datetime → ISO 字符串"""
    if dt is None:
        return ""
    return dt.isoformat()


def _from_iso(s: str) -> Optional[datetime.datetime]:
    """ISO 字符串 → datetime"""
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


@dataclass
class ResearchState:
    """单次研究的全部状态（可序列化）"""

    # ── 原始业务字段 ──
    mode: ResearchMode = ResearchMode.STANDARD
    topic: str = ""
    questions: List[str] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    evidences: List[Evidence] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)

    # ── 过渡字段（保留不动） ──
    raw_searches: List[str] = field(default_factory=list)
    raw_analysis: str = ""
    report: str = ""
    citation_audit: str = ""
    review_comments: str = ""

    # ── 新增 Harness 字段（Milestone 1） ──
    task_id: str = ""                       # 任务唯一 ID
    current_node: str = ""                  # 当前正在执行的节点名
    completed_nodes: List[str] = field(default_factory=list)  # 已完成的节点列表
    failed_node: str = ""                   # 失败节点名（空=未失败）
    status: str = "created"                 # created | running | completed | failed
    updated_at: str = ""                     # ISO 格式的最后更新时间

    # ── 流程游标字段（Milestone 1 — 恢复用） ──
    current_step: str = ""                  # 当前唯一步骤名
    completed_steps: List[str] = field(default_factory=list)  # 已完成的步骤名列表
    audit_passed: bool = True               # 最近一次 audit 是否通过（恢复时用于判断是否需要 rewrite）

    # ── Deep 模式字段（Milestone 3） ──
    deep_workers: List[DeepWorkerState] = field(default_factory=list)
    deep_workers_completed: bool = False

    # ==================== 序列化 ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        将 ResearchState 转为可 JSON 序列化的 dict

        自动递归转换嵌套 dataclass（Source/Document/Evidence/Claim/Conflict）
        和 Enum 类型（ResearchMode → "fast"/"standard"/"deep"）。

        返回: 纯 dict，不含 dataclass/Enum 对象
        """
        # asdict 自动递归所有嵌套 dataclass、将 str Enum 转为值
        raw: dict = asdict(self)

        # updated_at 已为字符串，不需要额外处理
        return raw

    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchState":
        """
        从 dict 重建 ResearchState

        自动处理:
        - str → ResearchMode (mode 字段)
        - dict → Source/Document/Evidence/Claim/Conflict (列表字段)
        - str → datetime
        """
        d = dict(data)  # 不修改入参

        # ── mode: str → Enum ──
        mode_val = d.pop("mode", ResearchMode.STANDARD)
        if isinstance(mode_val, str):
            d["mode"] = ResearchMode(mode_val)
        elif isinstance(mode_val, ResearchMode):
            d["mode"] = mode_val

        # ── 嵌套 dataclass 列表: dict → object ──
        for field_name, cls_type in _NESTED_TYPES.items():
            val = d.pop(field_name, None)
            if isinstance(val, list):
                d[field_name] = _reconstruct_list(val, cls_type)
            else:
                d[field_name] = []

        # ── deep_workers: 特殊处理（内部还有嵌套） ──
        dw_val = d.pop("deep_workers", None)
        if isinstance(dw_val, list):
            d["deep_workers"] = [_reconstruct_deep_worker(item) if isinstance(item, dict) else item
                                 for item in dw_val]
        else:
            d["deep_workers"] = []

        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "ResearchState":
        """从 JSON 字符串重建 ResearchState"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, filepath: str) -> "ResearchState":
        """从 JSON 文件重建 ResearchState"""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ==================== 便捷方法 ====================

    def mark_node_start(self, node_name: str):
        """标记节点开始执行"""
        self.current_node = node_name
        self.status = "running"
        self.updated_at = _to_iso(datetime.datetime.now())

    def mark_node_end(self, node_name: str):
        """标记节点完成"""
        if node_name not in self.completed_nodes:
            self.completed_nodes.append(node_name)
        if self.current_node == node_name:
            self.current_node = ""
        self.updated_at = _to_iso(datetime.datetime.now())

    def mark_node_failed(self, node_name: str):
        """标记节点失败"""
        self.failed_node = node_name
        self.status = "failed"
        self.updated_at = _to_iso(datetime.datetime.now())

    # ── 流程游标 ──

    def mark_step_start(self, step_name: str):
        """标记一个唯一步骤开始"""
        self.current_step = step_name
        self.current_node = step_name
        self.status = "running"
        self.updated_at = _to_iso(datetime.datetime.now())

    def mark_step_end(self, step_name: str):
        """标记一个唯一步骤完成"""
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)
        # 也向 completed_nodes 写入（确保向后兼容）
        if step_name not in self.completed_nodes:
            self.completed_nodes.append(step_name)
        # 只清 current_step，不清 current_node（兼容旧逻辑）
        self.current_step = ""
        if self.current_node == step_name:
            self.current_node = ""
        self.updated_at = _to_iso(datetime.datetime.now())

    def mark_step_failed(self, step_name: str):
        """标记一个唯一步骤失败"""
        self.failed_node = step_name
        self.current_step = ""
        self.status = "failed"
        self.updated_at = _to_iso(datetime.datetime.now())
