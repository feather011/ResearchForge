"""
ClaimVerificationNode — Claim 语义验证节点

在 Synthesis 之后、Write 之前执行。
检查每条 Claim 的文本是否被其引用的 evidence_ids 实际支持。

防止问题：
1. LLM 生成没有证据支持的 Claim（过度推断）
2. evidence_ids 锚定的证据片段语义上不支撑结论文本

输出 VerifiedClaim 列表，ResearchGraph 据此设置 Claim.confidence。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
from ..orchestration.research_state import ResearchState, Claim
from ..core.react_engine import LLMProvider


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"             # 证据充分支持结论
    PARTIALLY_SUPPORTED = "partially"   # 部分支持，有推断成分
    UNSUPPORTED = "unsupported"         # 证据不支持或过度推断
    UNVERIFIED = "unverified"           # LLM 调用失败，未完成验证


@dataclass
class VerifiedClaim:
    """带验证状态的 Claim"""
    claim_index: int
    status: ClaimStatus
    explanation: str = ""


def run_claim_verification_node(
    state: ResearchState,
    llm: LLMProvider,
) -> List[VerifiedClaim]:
    """
    语义验证所有 Claim

    对每条有 evidence_ids 的 Claim，把结论文本 + 对应证据原文给 LLM，
    判断 evidence 是否真的支持 claim。

    会写回 Claim.confidence:
      SUPPORTED   → 1.0
      PARTIALLY   → 0.5
      UNSUPPORTED → 0.0
    """
    if not state.claims:
        return []

    # 建立 evidence_id → text 的查找表
    evidence_map = {ev.id: ev.text for ev in state.evidences}
    results: List[VerifiedClaim] = []

    for i, claim in enumerate(state.claims):
        if not claim.evidence_ids:
            results.append(VerifiedClaim(
                claim_index=i,
                status=ClaimStatus.UNSUPPORTED,
                explanation="未引用任何证据",
            ))
            continue

        # 收集引用的证据文本
        ev_texts = []
        for eid in claim.evidence_ids:
            txt = evidence_map.get(eid)
            if txt:
                ev_texts.append(f"[{eid}] {txt[:300]}")
            else:
                ev_texts.append(f"[{eid}] (证据不存在)")

        # 用 LLM 判断是否支持（所有 Claim 拼在一次调用里）
        # 但每条单独调 LLM 太贵，一次批量验证
        results.append(VerifiedClaim(
            claim_index=i,
            status=ClaimStatus.SUPPORTED,
            explanation="证据ID存在，需语义验证",
        ))

    # 一次 LLM 调用批量验证所有 Claim
    batch_results, unresolved_ids = _batch_verify(claims=state.claims, evidence_map=evidence_map, llm=llm)

    # 记录未解析的 Claim 到 metadata
    if unresolved_ids:
        meta = getattr(state, "metadata", None)
        if meta is None:
            meta = {}
            state.metadata = meta
        meta.setdefault("claim_verify_unresolved", [])
        meta["claim_verify_unresolved"].extend(unresolved_ids)
        meta["claim_verify_unresolved_count"] = len(unresolved_ids)

    # 合并结果
    for vc in batch_results:
        if 0 <= vc.claim_index < len(state.claims):
            # 写回 confidence
            if vc.status == ClaimStatus.SUPPORTED:
                state.claims[vc.claim_index].confidence = 1.0
            elif vc.status == ClaimStatus.PARTIALLY_SUPPORTED:
                state.claims[vc.claim_index].confidence = 0.5
            else:
                # UNSUPPORTED / UNVERIFIED 均设为 0.0
                state.claims[vc.claim_index].confidence = 0.0

    return batch_results


def _batch_verify(
    claims: List[Claim],
    evidence_map: dict,
    llm: LLMProvider,
) -> Tuple[List[VerifiedClaim], List[int]]:
    """一次 LLM 调用，批量验证所有 Claim"""
    if not claims:
        return [], []

    # 构造带编号的 evidence 参考表
    all_ev_texts = []
    seen_eids = set()
    for claim in claims:
        for eid in claim.evidence_ids:
            if eid not in seen_eids and eid in evidence_map:
                seen_eids.add(eid)
                all_ev_texts.append(f"[{eid}] {evidence_map[eid][:300]}")

    ev_context = "\n\n".join(all_ev_texts) if all_ev_texts else "(无可用证据)"

    # 构造每个 Claim 的验证请求
    claim_blocks = []
    for i, c in enumerate(claims):
        eids_str = ", ".join(c.evidence_ids) if c.evidence_ids else "(无)"
        claim_blocks.append(
            f"Claim{i}: 文本={c.text[:200]}\n"
            f"  引用证据: {eids_str}"
        )

    claims_text = "\n\n".join(claim_blocks)

    prompt = f"""你是一个证据验证专家。检查每条结论（Claim）是否被其引用的证据（Evidence）实际支持。

【可用的证据原文】
{ev_context}

【待验证的结论】
{claims_text}

对每条结论，判断其引用证据是否在语义上支持该结论。

输出格式（每行一条，严格按此格式）：
ClaimX: supported | partially | unsupported | 原因说明

规则：
- supported = 证据原文明确支持该结论，没有过度推断
- partially = 证据部分支持，但结论包含推断成分或额外信息
- unsupported = 证据不支持该结论，或结论严重过度推断

示例：
Claim0: supported | 证据中已明确提及该结论的相关数据
Claim1: partially | 证据支持主要观点，但结论中的因果关系是推断的
Claim2: unsupported | 证据未提及该结论的任何信息"""

    try:
        result = llm.generate(prompt)
    except Exception:
        # 不吞噬异常，由调用方（ResearchGraph）决定重试还是降级
        raise

    # 解析结果
    verified = {}
    for line in result.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        # 匹配 ClaimX: status | explanation
        parts = line.split(":", 1)
        label = parts[0].strip()
        if not label.startswith("Claim"):
            continue
        try:
            idx = int(label[5:])
        except ValueError:
            continue

        rest = parts[1].strip()
        status_str = ""
        explanation = rest

        if "|" in rest:
            status_str = rest.split("|")[0].strip().lower()
            explanation = "|".join(rest.split("|")[1:]).strip()
        else:
            # 没有 | 分隔，尝试从首词判断
            words = rest.split()
            if words:
                first = words[0].lower().rstrip(",").rstrip(":")
                if first in ("supported", "partially", "unsupported"):
                    status_str = first
                    explanation = " ".join(words[1:]) if len(words) > 1 else ""
                else:
                    status_str = "unsupported"
                    explanation = rest

        if status_str == "supported" or status_str == "true":
            status = ClaimStatus.SUPPORTED
        elif status_str == "partially" or status_str == "partial":
            status = ClaimStatus.PARTIALLY_SUPPORTED
        else:
            status = ClaimStatus.UNSUPPORTED

        verified[idx] = VerifiedClaim(
            claim_index=idx,
            status=status,
            explanation=explanation[:200],
        )

    # 按原顺序返回，未解析的默认 UNVERIFIED
    unresolved_ids = [i for i in range(len(claims)) if i not in verified]
    result_list = [
        verified.get(i, VerifiedClaim(
            claim_index=i,
            status=ClaimStatus.UNVERIFIED,
            explanation="LLM返回中缺失该Claim的验证结果，标记为未验证",
        ))
        for i in range(len(claims))
    ]
    return result_list, unresolved_ids
