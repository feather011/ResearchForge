"""
WriteNode — 报告撰写节点

基于收集的证据和问题，调用 LLM 撰写研究报告
"""

from typing import List
from ..orchestration.research_state import ResearchState, Evidence
from ..core.react_engine import LLMProvider


def run_write_node(
    state: ResearchState,
    llm: LLMProvider,
    extra_instructions: str = "",
    mode: str = "standard",
) -> str:
    """
    撰写研究报告

    mode: fast → 精简（~300字）, standard → 适中（~800字）, deep → 详细（~1500字）
    """
    # 按模式选长度要求
    length_map = {
        "fast": "不少于300字，精简扼要，直接列出核心发现",
        "standard": "不少于800字，内容充实，结构完整",
        "deep": "不少于1500字，深入详细，多维度分析",
    }
    length_req = length_map.get(mode, "不少于800字")

    # 按来源分组
    from collections import defaultdict
    by_source = defaultdict(list)
    for ev in state.evidences:
        by_source[ev.source_id].append(ev.text)

    source_blocks = []
    for sid, texts in by_source.items():
        # sid 已经是 "来源1" 格式
        block = f"[{sid}]\n" + "\n".join(texts[:3])
        source_blocks.append(block)

    context = "\n\n".join(source_blocks)

    extra_section = ""
    if extra_instructions:
        extra_section = f"""
【修改要求】
{extra_instructions}

请针对上述问题修改以下报告，确保：
- 每条关键信息标注 [来源X]
- 无依据的推测替换为有证据支持的内容
- 覆盖所有研究问题"""

    prompt = f"""你是一个研究专家。基于以下材料撰写一篇{length_req}的研究报告。

【研究主题】
{state.topic}

【研究问题】
{chr(10).join(f'- {q}' for q in state.questions)}

【收集的证据】
{context}
{extra_section}

请撰写完整报告，包含以下章节：
1. 研究背景与概述
2. 核心发现与关键数据（每条标注 [来源X]）
3. 多维度详细分析
4. 结论与展望

要求：
- {length_req}
- 每条关键信息必须标注 [来源X]
- 内容充实，论据充分"""

    result = llm.generate(prompt)
    # Fast 模式不用补长
    if len(result) < 200:
        result += f"\n\n---\n*注：本次研究基于 {len(state.evidences)} 条证据撰写，因材料有限报告篇幅较短。*"
    return result
