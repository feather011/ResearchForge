"""
Research Graph - 研究状态机（三种模式通用）

状态流转（完整）：
CREATED → PLANNING → SEARCHING → FETCHING → EXTRACTING
  → EVALUATING → GAP_SEARCHING → SYNTHESIZING
  → WRITING → AUDITING → HUMAN_REVIEW → COMPLETED

Fast 模式跳过：EVALUATING, GAP_SEARCHING, AUDITING, HUMAN_REVIEW
Standard 模式：全部经过，GAP_SEARCHING 最多 1 轮
Deep 模式：全部经过，GAP_SEARCHING 最多 2 轮，多 Worker 并行
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import logging

from .mode_policy import ModePolicy, ResearchMode
from .research_state import ResearchState

logger = logging.getLogger("ResearchGraph")


class State(Enum):
    """研究状态（13 个状态，通用）"""
    CREATED = "created"
    PLANNING = "planning"
    SEARCHING = "searching"
    FETCHING = "fetching"
    EXTRACTING = "extracting"
    EVALUATING = "evaluating"
    GAP_SEARCHING = "gap_searching"
    SYNTHESIZING = "synthesizing"
    WRITING = "writing"
    AUDITING = "auditing"
    HUMAN_REVIEW = "human_review"
    COMPLETED = "completed"
    FAILED = "failed"


# 不同模式的状态执行路径
MODE_FLOW = {
    ResearchMode.FAST: [
        State.PLANNING, State.SEARCHING, State.FETCHING,
        State.EXTRACTING, State.WRITING, State.COMPLETED,
    ],
    ResearchMode.STANDARD: [
        State.PLANNING, State.SEARCHING, State.FETCHING,
        State.EXTRACTING, State.EVALUATING,
        State.WRITING, State.AUDITING, State.HUMAN_REVIEW, State.COMPLETED,
    ],
    ResearchMode.DEEP: [
        State.PLANNING, State.SEARCHING, State.FETCHING,
        State.EXTRACTING, State.EVALUATING,
        State.SYNTHESIZING, State.WRITING, State.AUDITING,
        State.HUMAN_REVIEW, State.COMPLETED,
    ],
}


class ResearchGraph:
    """
    研究状态机 — 驱动节点执行（三种模式通用）

    用法:
        graph = ResearchGraph(mode=ResearchMode.FAST)
        result = graph.execute("RLHF技术", llm=llm, progress_callback=cb)

    ResearchService 只负责创建 Graph 并调用 execute()，
    所有编排逻辑和状态推进都在 Graph 内部完成。
    """

    def __init__(self, mode: ResearchMode = ResearchMode.STANDARD):
        self.mode = mode
        self.policy = ModePolicy.for_mode(mode)
        self.state = State.CREATED
        self.rs: Optional[ResearchState] = None  # 研究数据

    def start(self, topic: str) -> "ResearchGraph":
        """开始研究"""
        self.rs = ResearchState(mode=self.mode, topic=topic)
        self.state = State.CREATED
        logger.info(f"研究开始: mode={self.mode.value}, topic={topic}")
        return self

    def _flow(self) -> List[State]:
        """获取当前模式的状态流"""
        return MODE_FLOW.get(self.mode, MODE_FLOW[ResearchMode.STANDARD])

    def get_next_state(self) -> Optional[State]:
        """获取当前状态的下一个状态（考虑模式差异）"""
        flow = self._flow()
        if self.state not in flow:
            return flow[0]  # 当前状态不在模式流中，从头开始
        for i, s in enumerate(flow):
            if s == self.state and i + 1 < len(flow):
                return flow[i + 1]
        return None

    def advance(self) -> Optional[State]:
        """推进到下一个状态"""
        next_state = self.get_next_state()
        if next_state:
            self.state = next_state
            logger.info(f"状态推进: {self.state.value}")
        return next_state

    def complete(self):
        """COMPLETED: 完成"""
        self.state = State.COMPLETED
        logger.info("研究完成")

    # ==================== 主执行引擎 ====================

    def execute(
        self,
        topic: str,
        llm: "LLMProvider",
        progress_callback: Optional[Callable] = None,
        tracer: Optional["TraceCollector"] = None,
    ) -> Dict[str, Any]:
        """
        执行一次研究的完整流程（Fast / Standard 模式）

        内部按状态机定义推进状态，调用对应节点函数，
        Service 不需要关心这些 —— 只需拿到最终结果。

        tracer: 可选 TraceCollector，记录节点执行开始/结束
        """
        import time as _time
        if False:
            from ..trace import TraceCollector  # noqa

        _start_time = _time.time()
        _audit_passed = True
        _rewritten = 0
        _audit_issues = []

        # 延迟导入避免循环依赖
        from ..nodes.plan_node import run_plan_node
        from ..nodes.search_node import run_search_node
        from ..nodes.fetch_node import run_fetch_node
        from ..nodes.extract_node import run_extract_node
        from ..nodes.synthesis_node import run_synthesis_node
        from ..nodes.claim_verification_node import run_claim_verification_node
        from ..nodes.coverage_node import run_coverage_node
        from ..nodes.gap_agent import run_evidence_gap_agent
        from ..nodes.write_node import run_write_node

        self.start(topic)

        # ── 1. PLANNING ──
        self.state = self._flow()[0]
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_start",
                           action="PLANNING", input=topic)
        if progress_callback:
            progress_callback("Planner", f"正在为「{topic}」制定研究计划...")
        self.rs.questions = run_plan_node(topic, llm)
        logger.info(f"规划完成: {len(self.rs.questions)} 个问题")
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_end",
                           action="PLANNING", result=f"{len(self.rs.questions)}个问题")
        if progress_callback:
            progress_callback("Planner", "计划完成，开始搜索...")

        # ── 2. SEARCHING ──
        self.state = self._flow()[1]
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_start",
                           action="SEARCHING", input=f"{len(self.rs.questions)}个问题")
        self.rs.sources = run_search_node(
            self.rs.questions,
            sources_per_question=self.policy.search_sources,
            use_real_search=True,
            llm=llm,
        )
        logger.info(f"搜索完成: {len(self.rs.sources)} 个来源")
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_end",
                           action="SEARCHING", result=f"{len(self.rs.sources)}个来源")
        if progress_callback:
            progress_callback(
                "SearchAgent",
                f"找到 {len(self.rs.sources)} 个来源，抓取正文...",
                extra_data={
                    "sources": [
                        {"id": s.id, "title": s.title, "snippet": s.snippet}
                        for s in self.rs.sources
                    ]
                },
            )

        # ── 3. FETCHING ──
        self.state = self._flow()[2]
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_start",
                           action="FETCHING", input=f"{len(self.rs.sources)}个来源")
        self.rs.documents = run_fetch_node(
            self.rs.sources, max_pages=self.policy.search_sources
        )
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_end",
                           action="FETCHING", result=f"{len(self.rs.documents)}篇文档")
        if progress_callback:
            progress_callback("System", "提取证据片段...")

        # ── 4. EXTRACTING ──
        self.state = self._flow()[3]
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_start",
                           action="EXTRACTING", input=f"{len(self.rs.documents)}篇文档")
        self.rs.evidences = run_extract_node(self.rs.documents, self.rs.questions)
        logger.info(f"提取完成: {len(self.rs.evidences)} 条证据")
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_end",
                           action="EXTRACTING", result=f"{len(self.rs.evidences)}条证据")

        # ── 5. SYNTHESIS（所有模式都做） ──
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_start",
                           action="SYNTHESIS", input=f"{len(self.rs.evidences)}条证据")
        if progress_callback:
            progress_callback("AnalystAgent", "综合分析所有证据...")
        self.rs.claims = run_synthesis_node(self.rs, llm)
        logger.info(f"分析完成: {len(self.rs.claims)} 条核心结论")
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_end",
                           action="SYNTHESIS", result=f"{len(self.rs.claims)}条结论")

        # ── 5b. CLAIM_VERIFICATION（所有模式都做，验证后写回 confidence） ──
        if self.rs.claims:
            if tracer:
                tracer.record(agent_name="ResearchGraph", stage="node_start",
                               action="CLAIM_VERIFICATION",
                               input=f"验证{len(self.rs.claims)}条结论的证据支持")
            if progress_callback:
                progress_callback("AnalystAgent", "验证结论的证据支撑...")
            verified = run_claim_verification_node(self.rs, llm)
            unsupported = [v for v in verified if v.status.value == "unsupported"]
            if unsupported:
                logger.warning(f"发现 {len(unsupported)} 条无证据支持的结论")
                for v in unsupported:
                    logger.warning(f"  Claim{v.claim_index}: {self.rs.claims[v.claim_index].text[:60]}")
            if tracer:
                statuses = ", ".join(f"{v.claim_index}:{v.status.value}" for v in verified)
                tracer.record(agent_name="ResearchGraph", stage="node_end",
                               action="CLAIM_VERIFICATION", result=statuses)

        # ── 6. COVERAGE（仅在 Standard/Deep 启用） ──
        if self.policy.enable_coverage_check:
            self.state = State.EVALUATING
            if tracer:
                tracer.record(agent_name="ResearchGraph", stage="node_start",
                               action="COVERAGE_CHECK",
                               input=f"{len(self.rs.evidences)}条证据, {len(self.rs.questions)}个问题")
            if progress_callback:
                progress_callback("System", "检查证据完整性...")
            complete, gaps = run_coverage_node(self.rs)
            if tracer:
                tracer.record(agent_name="ResearchGraph", stage="node_end",
                               action="COVERAGE_CHECK",
                               result=f"{'完整' if complete else f'发现{len(gaps)}个缺口'}")
            if not complete and self.policy.max_gap_search_rounds > 0:
                if progress_callback:
                    progress_callback(
                        "System",
                        f"发现 {len(gaps)} 个缺口，启动补搜 Agent...",
                    )

                filled, filled_gaps = run_evidence_gap_agent(
                    self.rs,
                    gaps,
                    llm,
                    max_rounds=self.policy.max_gap_search_rounds,
                    progress_callback=progress_callback,
                    tracer=tracer,
                )
                if filled:
                    logger.info(
                        f"补搜完成: 填补了 {len(filled_gaps)}/{len(gaps)} 个缺口"
                    )
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_start",
                                       action="RE_SYNTHESIS",
                                       input=f"补搜后重新分析")
                    if progress_callback:
                        progress_callback("AnalystAgent", "重新综合分析...")
                    self.rs.claims = run_synthesis_node(self.rs, llm)
                    # 补搜后重新验证 Claim
                    if self.rs.claims:
                        run_claim_verification_node(self.rs, llm)
                    # 重新覆盖检查（确认缺口已填）
                    complete, gaps = run_coverage_node(self.rs)
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_end",
                                       action="RE_SYNTHESIS",
                                       result=f"{'缺口已填' if complete else '仍有缺'}")
                else:
                    logger.info("补搜未填补任何缺口")
            if progress_callback:
                progress_callback("System", "证据检查完成")

        # ── 7. WRITING ──
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_start",
                           action="WRITING",
                           input=f"{len(self.rs.evidences)}条证据, {len(self.rs.claims)}条结论")
        if progress_callback:
            progress_callback("WriterAgent", "撰写报告...")
        self.state = State.WRITING
        self.rs.report = run_write_node(self.rs, llm)
        logger.info(f"写作完成: {len(self.rs.report)} 字")
        if tracer:
            tracer.record(agent_name="ResearchGraph", stage="node_end",
                           action="WRITING", result=f"{len(self.rs.report)}字")

        # ── 8. AUDIT（仅在启用审计的模式做） ──
        if self.policy.enable_report_audit:
            from ..nodes.audit_node import run_audit_node

            self.state = State.AUDITING
            if tracer:
                tracer.record(agent_name="ResearchGraph", stage="node_start",
                               action="AUDIT", input=f"{len(self.rs.report)}字报告")
            if progress_callback:
                progress_callback("System", "审计报告质量...")

            audit = run_audit_node(self.rs, llm)
            _audit_passed = audit.passed
            _audit_issues = audit.issues[:5]  # 保留前5个用于分析
            if audit.passed:
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="AUDIT", result="审计通过")
                logger.info("审计通过")
            else:
                logger.warning(f"审计发现问题: {audit.issues}")
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="AUDIT",
                                   result=f"发现{len(audit.issues)}个问题",
                                   observation="; ".join(audit.issues[:3]))
                if self.policy.max_rewrite_rounds > 0 and audit.suggestions:
                    if progress_callback:
                        progress_callback("WriterAgent",
                            f"审计发现 {len(audit.issues)} 个问题，重写报告...")
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_start",
                                       action="REWRITE", observation=audit.suggestions[:300])
                    self.rs.report = run_write_node(
                        self.rs, llm, extra_instructions=audit.suggestions
                    )
                    logger.info(f"重写完成: {len(self.rs.report)} 字")
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_end",
                                       action="REWRITE", result=f"{len(self.rs.report)}字")
                    _rewritten += 1
                    # 重写后重新审计
                    audit = run_audit_node(self.rs, llm)
                    _audit_passed = audit.passed
                    _audit_issues = audit.issues[:5]
                    if audit.passed:
                        logger.info("重写后审计通过")
                    else:
                        logger.warning(f"重写后审计仍发现 {len(audit.issues)} 个问题")
                    if progress_callback:
                        progress_callback("WriterAgent", "报告重写完成")

        # ── 9. COMPLETE ──
        self.complete()
        if progress_callback:
            progress_callback("System", "研究完成")

        # 统计 Claim 验证结果
        supported = sum(1 for c in self.rs.claims if c.confidence >= 0.9)
        unsupported = sum(1 for c in self.rs.claims if c.confidence == 0.0)

        return {
            "topic": self.rs.topic,
            "report": self.rs.report,
            "mode": self.mode.value,
            "stats": {
                "sources": len(self.rs.sources),
                "documents": len(self.rs.documents),
                "evidences": len(self.rs.evidences),
                "report_length": len(self.rs.report),
            },
            "tracer": tracer.get_all() if tracer else [],
            "claim_verification": {
                "total": len(self.rs.claims),
                "supported": supported,
                "unsupported": unsupported,
            },
            "audit": {
                "passed": _audit_passed,
                "rewritten": _rewritten,
                "issues": _audit_issues,
            },
            "_duration_s": round(_time.time() - _start_time, 2),
        }

    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "mode": self.mode.value,
            "state": self.state.value,
            "topic": self.rs.topic if self.rs else "",
            "sources": len(self.rs.sources) if self.rs else 0,
            "documents": len(self.rs.documents) if self.rs else 0,
            "evidences": len(self.rs.evidences) if self.rs else 0,
            "max_gap_rounds": self.policy.max_gap_search_rounds,
        }
