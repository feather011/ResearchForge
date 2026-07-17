"""
CoverageNode — 证据完整性检查节点

检查收集的证据能否回答所有研究问题，识别缺口
"""

from typing import List, Tuple
from ..orchestration.research_state import ResearchState
from ..orchestration.gap_replanner import GapReplanner


def run_coverage_node(state: ResearchState) -> Tuple[bool, List[str]]:
    """
    检查证据完整性

    原理：
    1. 对每个研究问题，检查是否有足够证据覆盖
    2. 证据覆盖标准：证据文本中至少包含问题的 2 个关键词
    3. 返回是否有缺口 + 缺口问题列表

    返回: (is_complete: bool, gaps: List[str])
    """
    replanner = GapReplanner()
    gaps = replanner.identify_gaps(state)
    return (len(gaps) == 0, gaps)
