"""
SynthesisNode — 综合分析节点

基于所有证据进行综合分析，生成核心结论。
每条结论绑定具体的 evidence_id，形成完整的追溯链。

追溯链：
  Source → Document → Evidence → Claim → Report
"""

import re
from typing import List
from ..orchestration.research_state import ResearchState, Claim
from ..core.react_engine import LLMProvider


def run_synthesis_node(state: ResearchState, llm: LLMProvider) -> List[Claim]:
    """
    综合分析所有证据，生成核心结论

    每条结论带有 evidence_ids，指向支持它的证据片段。
    解析后会验证每个 evidence_id 是否真实存在于 state.evidences 中。

    输入: ResearchState（含 evidences, questions）
    输出: List[Claim]（核心结论列表，每条绑定 evidence_ids）
    """
    if not state.evidences:
        return []

    # 收集所有有效证据 ID，用于后期验证
    valid_ids = {ev.id for ev in state.evidences}

    # 把每条证据编号，让 LLM 引用具体 ID
    evidence_blocks = []
    for ev in state.evidences:
        evidence_blocks.append(
            f"[{ev.id}] 来源: {ev.source_id} | 内容: {ev.text[:300]}"
        )

    context = "\n\n".join(evidence_blocks)

    prompt = f"""你是一个分析师。基于以下证据进行综合分析，生成核心结论。

【研究问题】
{chr(10).join(f'- {q}' for q in state.questions)}

【收集的证据（编号格式：<证据ID> → 内容）】
{context}

请提供核心结论（3-5条）。

输出格式（每行一条）：
[证据ID1, 证据ID2, ...] 结论文本

示例：
[ev_0, ev_2] RLHF 的核心思想是通过人类反馈来优化语言模型，使其输出更符合人类偏好
[ev_1] Anthropic 提出的 RLHF 变体使用"偏好模型"来替代人类直接评分

要求：
- 每条结论必须标注使用的证据 ID（在方括号中，多个用逗号分隔）
- 每条结论至少引用 1 条证据，最多引用 3 条
- 证据 ID 必须是上方给出的 ID（如 ev_0, gap_ev_1）
- 结论文本简洁明确"""
    result = llm.generate(prompt)

    # 解析行首 [ev_0, gap_ev_1] 文本 → Claim
    claims = []
    pattern = re.compile(r'^\s*\[([^\]]+)\]\s*(.+)')

    for line in result.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            continue

        ids_str = m.group(1)
        text = m.group(2).strip()

        # 提取 ID：按逗号分隔，每个 strip 后保留完整 ID
        parsed_ids = [pid.strip() for pid in ids_str.split(",")]

        # 只保留确实存在于证据列表中的 ID
        valid_parsed = [pid for pid in parsed_ids if pid in valid_ids]

        if not valid_parsed:
            continue

        text = text[:200]
        claims.append(Claim(text=text, evidence_ids=valid_parsed))

    # 兜底：如果 parsing 全失败，取第一条有效行的前 200 字
    if not claims:
        first_line = result.strip().split("\n")[0][:200]
        if first_line:
            claims.append(Claim(text=first_line))

    return claims
