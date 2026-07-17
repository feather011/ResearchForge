"""
EvidenceGapAgent — 证据缺口补搜 Agent

在 CoverageNode 发现证据缺口后触发。
使用 ReActAgent 循环，决定搜索什么、是否抓取、是否找到足够证据。

流程:
  CoverageNode 发现缺口 → EvidenceGapAgent.run(gaps)
    → Think: 分析缺什么信息
    → Act: 搜索 / 抓取 / 查证据  (ReAct 循环, 最多 3 轮)
    → Observe: 检查是否补全了缺口
    → 完成 / 继续循环
"""

import logging
from typing import List

from ..core.react_engine import ReActAgent, BaseTool, LLMProvider
from ..orchestration.research_state import ResearchState, Source, Document, Evidence
from ..tools.search import WebSearchTool
from ..rag.evidence import EvidenceRetriever

logger = logging.getLogger("EvidenceGapAgent")


class ToolRegistry:
    """工具注册中心 — 统一管理 Agent 可用的工具"""

    def __init__(self):
        self._tools: dict = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_all(self) -> list:
        return list(self._tools.values())

    def get(self, name: str) -> BaseTool:
        return self._tools.get(name)


class GapSearchTool(BaseTool):
    """搜索工具 — 针对缺口问题搜索网页"""

    def __init__(self, state: ResearchState):
        self.state = state
        self._searcher = WebSearchTool(use_real_search=True)

    @property
    def name(self) -> str:
        return "gap_search"

    @property
    def description(self) -> str:
        return "搜索指定问题的最新信息。输入: query(搜索关键词)"

    def run(self, query: str, **kwargs) -> str:
        result = self._searcher.run(query)
        self.state.sources.append(Source(
            id=f"gap_{len(self.state.sources)+1}",
            snippet=result[:300],
            title=query[:60],
        ))
        return result[:500]


class GapFetchTool(BaseTool):
    """抓取工具 — 获取搜索结果中的详细内容"""

    def __init__(self, state: ResearchState):
        self.state = state

    @property
    def name(self) -> str:
        return "gap_fetch"

    @property
    def description(self) -> str:
        return "抓取指定来源的详细内容。输入: source_id(来源ID)"

    def run(self, source_id: str, **kwargs) -> str:
        for s in self.state.sources:
            if s.id == source_id:
                from ..nodes.fetch_node import run_fetch_node
                docs = run_fetch_node([s], max_pages=1)
                if docs:
                    self.state.documents.extend(docs)
                    return docs[0].content[:500]
                return "抓取失败"
        return f"未找到来源: {source_id}"


class GapEvidenceSearchTool(BaseTool):
    """证据检索工具 — 检查当前文档中是否有相关证据"""

    def __init__(self, state: ResearchState):
        self.state = state

    @property
    def name(self) -> str:
        return "evidence_search"

    @property
    def description(self) -> str:
        return "在已有文档中检索与问题相关的证据。输入: question(问题)"

    def run(self, question: str, **kwargs) -> str:
        if not self.state.documents:
            return "当前没有文档可供检索"
        retriever = EvidenceRetriever()
        retriever.add_documents(self.state.documents)
        results = retriever.retrieve(question, top_k=3)
        if results:
            texts = [r["text"][:200] for r in results]
            return "\n\n".join(texts)
        return f"未找到与「{question}」相关的证据"


def run_evidence_gap_agent(
    state: ResearchState,
    gaps: List[str],
    llm: LLMProvider,
    max_rounds: int = 3,
    progress_callback=None,
    tracer=None,
):
    """
    运行证据缺口补搜 Agent

    在 CoverageNode 发现缺口后调用。
    用 ReActAgent 循环搜索/抓取/检索，直到补全缺口或达到上限。

    参数:
      state: 当前研究状态（直接修改 sources/documents/evidences）
      gaps: 需要补搜的问题列表
      llm: LLM 实例
      max_rounds: 最大循环次数
      progress_callback: 进度回调

    返回: (filled: bool, filled_gaps: List[str])
    """
    query = "需要补充以下问题的信息：\n" + "\n".join(f"- {g}" for g in gaps)

    registry = ToolRegistry()
    registry.register(GapSearchTool(state))
    registry.register(GapFetchTool(state))
    registry.register(GapEvidenceSearchTool(state))

    agent = ReActAgent(
        tools=registry.get_all(),
        llm=llm,
        max_steps=max_rounds,
    )

    if progress_callback:
        progress_callback("GapAgent", f"证据缺口补搜启动: {len(gaps)} 个缺口")

    if tracer:
        tracer.record(agent_name="GapAgent", stage="think",
                       action="gap_agent_start",
                       input=f"缺口数: {len(gaps)}, 最大轮次: {max_rounds}",
                       result="; ".join(gaps[:3]))

    result = agent.run(query, tracer=tracer)

    if progress_callback:
        progress_callback("GapAgent", f"补搜完成: {'成功' if result.success else '部分完成'}")

    # 检查哪些缺口被补上了
    filled_gaps = []
    for gap in gaps:
        retriever = EvidenceRetriever()
        retriever.add_documents(state.documents)
        hits = retriever.retrieve(gap, top_k=1)
        if hits:
            filled_gaps.append(gap)
            state.evidences.append(Evidence(
                id=f"gap_ev_{len(state.evidences)}",
                source_id=hits[0]["source_id"],
                text=hits[0]["text"],
            ))

    return len(filled_gaps) > 0, filled_gaps
