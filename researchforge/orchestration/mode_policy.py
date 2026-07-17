"""
三种运行模式策略控制

定义 ResearchMode 枚举和 ModePolicy 策略类。
根据模式决定研究流程中执行哪些节点、搜索多少来源、是否补搜等。
"""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field


class ResearchMode(str, Enum):
    """运行模式"""
    FAST = "fast"           # 快速：无补搜、无审核、短报告
    STANDARD = "standard"   # 标准：一轮补搜、引用审核、人工审核
    DEEP = "deep"           # 深度：多Worker、多轮补搜、冲突分析


@dataclass
class ModePolicy:
    """模式策略：决定研究行为"""
    mode: ResearchMode = ResearchMode.STANDARD

    # ── 搜索 ──
    search_sources: int = 5           # 每个问题搜索几个来源
    max_fetch_pages: int = 3          # 抓取几个网页正文

    # ── 补搜 ──
    enable_coverage_check: bool = True    # 是否检查证据完整性
    max_gap_search_rounds: int = 1        # 最多补搜几轮

    # ── 多Worker ──
    enable_multi_worker: bool = False     # 是否启用多Worker并行
    workers_count: int = 3                # Worker数量（Deep模式）

    # ── 冲突分析 ──
    enable_conflict_analysis: bool = False

    # ── 报告 ──
    report_length: str = "medium"         # short / medium / long

    # ── 报告审计 ──
    enable_report_audit: bool = True      # 是否审计报告质量
    max_rewrite_rounds: int = 1           # 最大重写次数

    # ── 人工审核 ──
    require_human_review: bool = True     # 是否进入人工审核阶段

    @classmethod
    def for_mode(cls, mode: ResearchMode) -> "ModePolicy":
        """根据模式创建对应策略"""
        configs = {
            ResearchMode.FAST: dict(
                search_sources=3,
                max_fetch_pages=1,
                enable_coverage_check=False,
                max_gap_search_rounds=0,
                enable_multi_worker=False,
                enable_conflict_analysis=False,
                report_length="short",
                enable_report_audit=False,
                max_rewrite_rounds=0,
                require_human_review=False,
            ),
            ResearchMode.STANDARD: dict(
                search_sources=5,
                max_fetch_pages=3,
                enable_coverage_check=True,
                max_gap_search_rounds=1,
                enable_multi_worker=False,
                enable_conflict_analysis=False,
                report_length="medium",
                enable_report_audit=True,
                max_rewrite_rounds=1,
                require_human_review=True,
            ),
            ResearchMode.DEEP: dict(
                search_sources=8,
                max_fetch_pages=5,
                enable_coverage_check=True,
                max_gap_search_rounds=2,
                enable_multi_worker=True,
                workers_count=5,
                enable_conflict_analysis=True,
                report_length="long",
                enable_report_audit=True,
                max_rewrite_rounds=1,
                require_human_review=True,
            ),
        }
        return cls(mode=mode, **configs[mode])
