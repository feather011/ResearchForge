"""
MockProvider — 稳定的 Mock LLM、Search、Fetch 数据

用于无需真实 API Key 的 Demo 运行和测试。
不修改正式业务逻辑，所有 mock 代码仅在此文件和 demo 测试中使用。
"""

import time
from unittest.mock import Mock
from typing import List

from researchforge.orchestration.research_state import Source, Document, Evidence, Claim
from researchforge.nodes.claim_verification_node import VerifiedClaim, ClaimStatus


def create_mock_llm():
    """创建稳定的 Mock LLM Provider"""
    llm = Mock()

    def mock_generate(prompt: str, **kwargs) -> str:
        if "首席研究员" in prompt or "拆分为" in prompt:
            return "1. 核心原理与传统方法\n2. 关键技术与实现路径\n3. 最新进展与应用场景"
        if "研究助手" in prompt or "拆解" in prompt:
            return "1. 技术原理\n2. 应用案例\n3. 发展趋势"
        if "分析师" in prompt or "综合分析" in prompt:
            return ("[ev_0, ev_1] 该技术的核心原理基于注意力机制和并行计算架构\n"
                    "[ev_0] 主流实现方案包括优化内存访问模式和计算图重写\n"
                    "[ev_1] 实验结果表明该方法在保持精度的同时显著提升了推理速度")
        if "证据验证" in prompt or "Claim" in prompt:
            return "Claim0: supported | 证据明确支持该结论\nClaim1: supported | 证据充分\nClaim2: partially | 部分支持"
        if "报告质量" in prompt or "审计" in prompt or "审核员" in prompt:
            return "PASSED: 是\n报告结构完整，引用了足够的来源支持主要观点。"
        if "搜索查询" in prompt or "搜索短语" in prompt:
            return "1. 核心原理\n2. 技术实现"
        if "修改" in prompt.lower() or "rewrite" in prompt.lower():
            return "根据审核意见修改后的完整报告。\n\n[来源1] 和 [来源2] 都支持这个结论。"
        if "冲突" in prompt:
            return "无冲突"
        return "这是一个稳定的 Mock 回复，用于演示和测试。"

    llm.generate = mock_generate
    llm.generate.side_effect = mock_generate
    return llm


def create_mock_search():
    """创建稳定的 Mock Search 节点"""

    def mock_search(questions, **kw) -> List[Source]:
        return [
            Source(id="来源1", title="技术原理概述", snippet="该技术基于深度学习框架实现", url="https://example.com/1"),
            Source(id="来源2", title="最新研究进展", snippet="2024年多项研究证明了该方法的有效性", url="https://example.com/2"),
            Source(id="来源3", title="工程实践指南", snippet="在实际部署中需要注意内存和延迟的权衡", url="https://example.com/3"),
        ]

    return mock_search


def create_mock_fetch():
    """创建稳定的 Mock Fetch 节点"""

    def mock_fetch(sources, **kw) -> List[Document]:
        return [
            Document(content="该技术通过优化计算图和内存访问模式来提升性能。实验表明推理速度提升2-3倍。",
                     source_id=src.id, url=getattr(src, "url", ""), title=getattr(src, "title", ""))
            for src in (sources or [])[:2]
        ]

    return mock_fetch


def create_mock_extract():
    """创建稳定的 Mock Extract 节点"""

    def mock_extract(docs, questions, **kw) -> List[Evidence]:
        return [
            Evidence(id=f"ev_{i}", source_id=getattr(d, "source_id", "来源1"),
                     text=f"证据片段 {i}: 该技术的关键实现方式是{d.content[:50] if hasattr(d, 'content') else '标准实现'}")
            for i, d in enumerate(docs or [])
        ]

    return mock_extract


def create_mock_claim_verify():
    """创建稳定的 Mock Claim Verification（返回 VerifiedClaim 列表 + 空 unresolved）"""

    def mock_verify(state, llm) -> List[VerifiedClaim]:
        claims = state.claims if hasattr(state, "claims") else []
        results = []
        for i, c in enumerate(claims):
            results.append(VerifiedClaim(
                claim_index=i,
                status=ClaimStatus.SUPPORTED if c.confidence >= 0.9 else ClaimStatus.PARTIALLY_SUPPORTED,
                explanation="Mock 验证通过",
            ))
        return results

    return mock_verify


def setup_mock_patches(monkeypatch):
    """
    注入所有 Mock 节点。用于 Demo 运行和测试。

    注意：仅在 demo 和测试环境中调用，不影响正式业务逻辑。
    """
    from researchforge.nodes import (
        search_node, fetch_node, extract_node, synthesis_node,
        write_node, claim_verification_node, coverage_node, audit_node
    )

    monkeypatch.setattr(search_node, "run_search_node", create_mock_search())
    monkeypatch.setattr(fetch_node, "run_fetch_node", create_mock_fetch())
    monkeypatch.setattr(extract_node, "run_extract_node", create_mock_extract())
    monkeypatch.setattr(claim_verification_node, "run_claim_verification_node", create_mock_claim_verify())

    # Synthesis: return stable claims
    monkeypatch.setattr(synthesis_node, "run_synthesis_node",
                        lambda state, llm: [
                            Claim(text="该技术通过优化内存访问和计算图来提升性能", evidence_ids=["ev_0", "ev_1"]),
                            Claim(text="主流方案在保持精度的同时显著提升了推理速度", evidence_ids=["ev_0"]),
                        ])

    # Write: return stable report with valid citations
    monkeypatch.setattr(write_node, "run_write_node",
                        lambda state, llm=None, extra_instructions="", mode="fast":
                        "## 技术概述\n\n"
                        "该技术是近年来重要的研究方向。[来源1] 提出了核心实现方案，"
                        "[来源2] 则从理论上分析了其有效性。\n\n"
                        "## 核心发现\n\n"
                        "实验数据[来源1]表明该方法在标准测试集上取得了优异表现。"
                        "同时[来源3]也验证了其在生产环境中的可行性。\n\n"
                        "## 结论\n\n"
                        "该技术具有重要的应用价值和发展前景。")

    # Coverage: mark all questions as covered
    monkeypatch.setattr(coverage_node, "run_coverage_node",
                        lambda state: (True, []))

    # Audit: always pass
    monkeypatch.setattr(audit_node, "run_audit_node",
                        lambda state, llm: type("AuditResult", (), {
                            "passed": True, "issues": [], "suggestions": ""
                        })())

    return True
