"""
Claim Verification 缺省默认值修复测试
"""

from unittest.mock import Mock

import pytest
from researchforge.nodes.claim_verification_node import (
    ClaimStatus,
    VerifiedClaim,
    run_claim_verification_node,
)
from researchforge.orchestration.research_state import ResearchState, Claim, Evidence


def _make_state(claims, evidences=None, metadata=None):
    """创建带 claims 和 evidences 的 ResearchState"""
    state = ResearchState(topic="test")
    state.claims = claims
    state.evidences = evidences or []
    if metadata:
        state.metadata = metadata
    return state


class TestClaimVerificationDefault:
    """_batch_verify 缺失解析默认值测试"""

    def test_partial_claims_missing_defaults_to_unverified(self):
        """部分 Claim 缺失时默认 UNVERIFIED，不影响其他"""
        llm = Mock()
        # LLM 只返回 Claim0，不返回 Claim1
        llm.generate.return_value = "Claim0: supported | 证据已确认该结论"

        ev = Evidence(id="ev_0", source_id="s1", text="确认数据")
        c0 = Claim(text="结论1", evidence_ids=["ev_0"])
        c1 = Claim(text="结论2", evidence_ids=["ev_0"])
        state = _make_state(claims=[c0, c1], evidences=[ev])

        results = run_claim_verification_node(state, llm)

        assert len(results) == 2
        assert results[0].status == ClaimStatus.SUPPORTED, "Claim0 应保持 SUPPORTED"
        assert results[1].status == ClaimStatus.UNVERIFIED, "Claim1 应默认 UNVERIFIED"
        assert state.claims[0].confidence == 1.0
        assert state.claims[1].confidence == 0.0  # UNVERIFIED → 0.0

    def test_all_claims_missing_all_unverified(self):
        """全部 Claim 缺失时全部 UNVERIFIED"""
        llm = Mock()
        # LLM 返回完全无关内容，不解析出任何 Claim
        llm.generate.return_value = "我不理解这些证据"

        ev = Evidence(id="ev_0", source_id="s1", text="数据1")
        c0 = Claim(text="结论1", evidence_ids=["ev_0"])
        c1 = Claim(text="结论2", evidence_ids=["ev_0"])
        state = _make_state(claims=[c0, c1], evidences=[ev])

        results = run_claim_verification_node(state, llm)

        assert len(results) == 2
        assert all(r.status == ClaimStatus.UNVERIFIED for r in results), \
            f"全部应 UNVERIFIED, 实际 {[r.status for r in results]}"
        assert state.claims[0].confidence == 0.0
        assert state.claims[1].confidence == 0.0

    def test_normal_all_supported_unaffected(self):
        """正常状态不受影响——全部 SUPPORTED/PARTIALLY/UNSUPPORTED 保持不变"""
        llm = Mock()
        llm.generate.return_value = (
            "Claim0: supported | 证据支持\n"
            "Claim1: partially | 部分支持\n"
            "Claim2: unsupported | 不支持"
        )

        ev = Evidence(id="ev_0", source_id="s1", text="数据")
        c0 = Claim(text="结论1", evidence_ids=["ev_0"])
        c1 = Claim(text="结论2", evidence_ids=["ev_0"])
        c2 = Claim(text="结论3", evidence_ids=["ev_0"])
        state = _make_state(claims=[c0, c1, c2], evidences=[ev])

        results = run_claim_verification_node(state, llm)

        assert len(results) == 3
        assert results[0].status == ClaimStatus.SUPPORTED
        assert results[1].status == ClaimStatus.PARTIALLY_SUPPORTED
        assert results[2].status == ClaimStatus.UNSUPPORTED
        assert state.claims[0].confidence == 1.0
        assert state.claims[1].confidence == 0.5
        assert state.claims[2].confidence == 0.0

    def test_metadata_records_unresolved_count_and_ids(self):
        """缺失的 Claim 记录到 metadata"""
        llm = Mock()
        llm.generate.return_value = "Claim0: supported | 证据支持"

        ev = Evidence(id="ev_0", source_id="s1", text="数据")
        c0 = Claim(text="结论1", evidence_ids=["ev_0"])
        c1 = Claim(text="结论2", evidence_ids=["ev_0"])
        c2 = Claim(text="结论3", evidence_ids=["ev_0"])
        state = _make_state(claims=[c0, c1, c2], evidences=[ev])

        run_claim_verification_node(state, llm)

        meta = state.metadata
        assert meta.get("claim_verify_unresolved_count") == 2, \
            f"应记录 2 个缺失, 实际 {meta.get('claim_verify_unresolved_count')}"
        unresolved = meta.get("claim_verify_unresolved", [])
        assert 1 in unresolved
        assert 2 in unresolved
        assert 0 not in unresolved

    def test_metadata_empty_when_no_unresolved(self):
        """全部解析成功时不记录 metadata"""
        llm = Mock()
        llm.generate.return_value = "Claim0: supported | 证据支持"

        ev = Evidence(id="ev_0", source_id="s1", text="数据")
        c0 = Claim(text="结论1", evidence_ids=["ev_0"])
        state = _make_state(claims=[c0], evidences=[ev])

        run_claim_verification_node(state, llm)

        # metadata 可能未创建（无 unresolved 时不写入）
        meta = getattr(state, "metadata", {}) or {}
        assert meta.get("claim_verify_unresolved_count", 0) == 0
