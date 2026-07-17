"""
GapReplanner — 证据缺口检测和补搜规划

检查已收集的证据是否能回答所有研究问题。
识别缺口，生成补搜计划。
"""

from typing import List
from .research_state import ResearchState


class GapReplanner:
    """缺口检测与补搜规划"""

    def identify_gaps(self, state: ResearchState) -> List[str]:
        """
        识别证据缺口：哪些问题还没有足够的证据支持
        返回需要补搜的问题列表
        """
        if not state.questions or not state.evidences:
            return state.questions[:]  # 没有证据 → 全部缺口

        gaps = []
        for q in state.questions:
            q_words = set(q.lower().split())
            # 检查是否有证据覆盖这个问题
            covered = False
            for ev in state.evidences:
                ev_words = set(ev.text.lower().split() if ev.text else "")
                overlap = len(q_words & ev_words)
                if overlap >= min(2, len(q_words)):  # 至少 2 个词重叠
                    covered = True
                    break
            if not covered:
                gaps.append(q)

        return gaps

    def create_gap_plan(self, gaps: List[str]) -> List[str]:
        """为缺口生成补搜查询"""
        return gaps  # 直接用原问题补搜
