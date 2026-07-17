"""
ResearchService — 研究服务
把 ResearchGraph + 节点管线 + LLM 绑定成可调用接口
"""

import logging
from typing import Dict, Any, Optional, Callable

from .orchestration import ResearchGraph, ResearchMode, ResearchState, State
from .core.react_engine import LLMProvider
from .nodes.deep_research import run_deep_research
from .nodes.write_node import run_write_node

logger = logging.getLogger("ResearchService")


class ResearchService:
    """
    研究服务 — 统一研究入口

    用法:
        service = ResearchService(llm=llm)
        result = service.run("RLHF技术", mode=ResearchMode.FAST)

    ResearchService 只做两件事：
      1. 模式路由（Fast/Standard → ResearchGraph.execute, Deep → run_deep_research）
      2. 结果组装
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def run(
        self,
        topic: str,
        mode: ResearchMode = ResearchMode.FAST,
        progress_callback: Optional[Callable] = None,
        tracer=None,
    ) -> Dict[str, Any]:
        """执行一次研究"""
        if mode == ResearchMode.DEEP:
            return self._run_deep(topic, progress_callback, tracer=tracer)
        return self._run_standard(topic, mode, progress_callback, tracer=tracer)

    def _run_deep(self, topic: str, progress_callback, tracer=None) -> Dict[str, Any]:
        """Deep 模式：多 Worker 并行研究"""
        if progress_callback:
            progress_callback("LeadResearcher", f"制定深度研究计划...")

        result = run_deep_research(topic, self.llm, num_workers=3, tracer=tracer)

        if progress_callback:
            progress_callback("LeadResearcher",
                f"规划完成: {len(result['tasks'])}个子任务, 启动Worker...")

        all_sources = result["merged"]["sources"]
        if progress_callback and all_sources:
            progress_callback("SearchAgent",
                f"所有Worker共找到 {len(all_sources)} 个来源",
                extra_data={"sources": [
                    {"id": s.id, "title": s.title, "snippet": s.snippet}
                    for s in all_sources[:10]]})

        for wr in result["worker_results"]:
            if progress_callback:
                progress_callback(f"Worker {wr['worker_id']}",
                    f"完成: {len(wr['evidences'])}条证据")

        if progress_callback:
            progress_callback("AnalystAgent", "综合分析所有Worker结果...")
            progress_callback("LeadResearcher", "合并结果, 生成报告...")

        state = ResearchState(mode=ResearchMode.DEEP, topic=topic)
        state.questions = result["tasks"]
        state.sources = result["merged"]["sources"]
        state.documents = result["merged"]["documents"]
        state.evidences = result["merged"]["evidences"]
        state.conflicts = result["conflicts"]

        # 走 Claim 链：Synthesis → ClaimVerification
        from .nodes.synthesis_node import run_synthesis_node
        from .nodes.claim_verification_node import run_claim_verification_node

        state.claims = run_synthesis_node(state, self.llm)
        run_claim_verification_node(state, self.llm)

        # Coverage + GapAgent（Deep 模式启用证据完整性检查）
        from .nodes.coverage_node import run_coverage_node
        from .nodes.gap_agent import run_evidence_gap_agent

        if True:  # Deep 模式始终做覆盖检查
            if progress_callback:
                progress_callback("System", "检查证据完整性...")
            complete, gaps = run_coverage_node(state)
            if not complete:
                if progress_callback:
                    progress_callback("System", f"发现 {len(gaps)} 个缺口，启动补搜 Agent...")
                filled, filled_gaps = run_evidence_gap_agent(
                    state, gaps, self.llm,
                    max_rounds=2,  # Deep 模式多一轮
                    progress_callback=progress_callback,
                    tracer=tracer,
                )
                if filled:
                    logger.info(f"补搜完成: 填补了 {len(filled_gaps)}/{len(gaps)} 个缺口")
                    if progress_callback:
                        progress_callback("AnalystAgent", "补搜后重新分析...")
                    state.claims = run_synthesis_node(state, self.llm)
                    run_claim_verification_node(state, self.llm)

        # Write
        if progress_callback:
            progress_callback("WriterAgent", "撰写报告...")
        report = run_write_node(state, self.llm)
        state.report = report

        if result["conflicts"]:
            conflict_text = "\n".join(f"- {c.claim}" for c in result["conflicts"])
            report += f"\n\n---\n### 来源冲突\n{conflict_text}"

        # Audit + Rewrite
        _audit_passed = True
        _rewritten = 0
        _audit_issues = []
        from .nodes.audit_node import run_audit_node

        if progress_callback:
            progress_callback("System", "审计报告质量...")
        audit = run_audit_node(state, self.llm)
        _audit_passed = audit.passed
        _audit_issues = audit.issues[:5]
        if not audit.passed:
            logger.warning(f"Deep审计发现问题: {audit.issues}")
            if audit.suggestions:
                if progress_callback:
                    progress_callback("WriterAgent", f"审计发现 {len(audit.issues)} 个问题，重写...")
                state.report = run_write_node(state, self.llm, extra_instructions=audit.suggestions)
                _rewritten = 1
                # 重写后重新审计
                audit = run_audit_node(state, self.llm)
                _audit_passed = audit.passed
                _audit_issues = audit.issues[:5]
                if audit.passed:
                    logger.info("Deep重写后审计通过")
                else:
                    logger.warning(f"Deep重写后审计仍发现 {len(audit.issues)} 个问题")

        if progress_callback:
            progress_callback("System", "研究完成")

        supported = sum(1 for c in state.claims if c.confidence >= 0.9)
        unsupported = sum(1 for c in state.claims if c.confidence == 0.0)

        return {
            "topic": topic,
            "report": state.report,
            "mode": "deep",
            "stats": {
                "sources": len(state.sources),
                "documents": len(state.documents),
                "evidences": len(state.evidences),
                "report_length": len(state.report),
                "workers": len(result["worker_results"]),
                "conflicts": len(result["conflicts"]),
            },
            "tracer": tracer.get_all() if tracer else [],
            "claim_verification": {
                "total": len(state.claims),
                "supported": supported,
                "unsupported": unsupported,
            },
            "audit": {
                "passed": _audit_passed,
                "rewritten": _rewritten,
                "issues": _audit_issues,
            },
            "require_human_review": True,
        }

    def _run_standard(
        self,
        topic: str,
        mode: ResearchMode,
        progress_callback: Optional[Callable] = None,
        tracer=None,
    ) -> Dict[str, Any]:
        """Fast / Standard 模式 — 委托给 ResearchGraph.execute()"""
        graph = ResearchGraph(mode=mode)
        result = graph.execute(topic, self.llm, progress_callback, tracer=tracer)
        # 是否进入人工审核阶段
        result["require_human_review"] = (
            mode != ResearchMode.FAST
            and graph.policy.require_human_review
        )
        return result
