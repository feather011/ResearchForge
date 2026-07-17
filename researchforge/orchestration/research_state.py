"""
ResearchState — 研究任务的数据中心

替代旧的 HybridMemory 作为单次研究的数据存储。
"""
from dataclasses import dataclass, field
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
class ResearchState:
    """单次研究的全部状态"""
    mode: ResearchMode = ResearchMode.STANDARD
    topic: str = ""
    questions: List[str] = field(default_factory=list)    # 子问题列表
    sources: List[Source] = field(default_factory=list)    # 搜索到的来源
    documents: List[Document] = field(default_factory=list) # 抓取的正文
    evidences: List[Evidence] = field(default_factory=list) # 提取的证据
    claims: List[Claim] = field(default_factory=list)       # 核心结论
    conflicts: List[Conflict] = field(default_factory=list) # 冲突

    # 原始搜索结果和分析文本（过渡用）
    raw_searches: List[str] = field(default_factory=list)
    raw_analysis: str = ""

    report: str = ""
    citation_audit: str = ""
    review_comments: str = ""
