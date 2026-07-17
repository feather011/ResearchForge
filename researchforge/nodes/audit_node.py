"""
ReportAuditNode — 报告质量审计节点

在 Write 之后执行，检查报告质量。
提供 Correction 信号，支持 Rewrite 闭环。

检查项：
1. 研究问题是否被报告覆盖
2. 报告中的引用 [来源X] 是否对应真实来源
3. 报告中是否存在无证据依据的论述（LLM 幻觉检测）
4. 报告结构是否完整

注意：Claim 级别的证据语义验证已移到 ClaimVerificationNode。
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple
from ..orchestration.research_state import ResearchState
from ..core.react_engine import LLMProvider


@dataclass
class AuditResult:
    """审计结果"""
    passed: bool
    issues: List[str] = field(default_factory=list)
    suggestions: str = ""


def run_audit_node(
    state: ResearchState,
    llm: LLMProvider,
) -> AuditResult:
    """
    对研究报告进行质量审计

    三个检查维度，全部通过才能 passed=True。
    """
    issues = []

    # ── 检查1：研究问题覆盖 ──
    issues += _check_question_coverage(state)

    # ── 检查2：引用完整性 ──
    issues += _check_citation_integrity(state)

    # ── 检查3：LLM 幻觉检测 + 结构质量 ──
    issues3, suggestions = _check_report_quality(state, llm)
    issues += issues3

    passed = len(issues) == 0
    suggestions_text = "\n".join(suggestions) if suggestions else ""

    return AuditResult(passed=passed, issues=issues, suggestions=suggestions_text)


def _check_question_coverage(state: ResearchState) -> List[str]:
    """检查1：每个研究问题是否在报告中被覆盖（不用LLM，关键词匹配判断）"""
    issues = []
    if not state.questions or not state.report:
        return []

    report_lower = state.report.lower()
    for qi, question in enumerate(state.questions):
        # 提取2字以上的段作为关键词（中文split自动按词/字切分，2+字段更靠谱）
        q_words = [w for w in question.lower().split() if len(w) > 1]
        if not q_words:
            continue
        # 只要有一个核心词出现在报告里就认为覆盖
        match_count = sum(1 for w in q_words if w in report_lower)
        if match_count == 0:
            issues.append(f"研究问题{qi+1}「{question[:60]}」在报告中未被充分覆盖")

    return issues


def _check_citation_integrity(state: ResearchState) -> List[str]:
    """检查2：报告中每个 [来源X] 是否对应真实存在的来源"""
    issues = []
    if not state.report:
        return []

    # 提取所有 [来源X] 标记
    citations = set(re.findall(r'\[来源\d+\]', state.report))
    if not citations:
        issues.append("报告中没有任何 [来源X] 引用标记")
        return issues

    # 验证每个来源 ID 是否存在
    valid_source_ids = {s.id for s in state.sources}
    for cite in citations:
        sid = cite.strip("[]")
        if sid not in valid_source_ids:
            issues.append(f"报告引用了不存在的来源：「{sid}」")

    return issues


def _check_report_quality(
    state: ResearchState, llm: LLMProvider
) -> Tuple[List[str], List[str]]:
    """检查3：用 LLM 检测报告质量（幻觉 + 结构 + 无依据内容）"""
    issues = []
    suggestions = []

    if not state.report or len(state.report) < 200:
        return issues, suggestions

    # 构造验证上下文：带上 Claim 的验证状态
    claim_summary_lines = []
    for i, c in enumerate(state.claims):
        conf_label = "高可信" if c.confidence >= 0.9 else "中可信" if c.confidence >= 0.5 else "低可信"
        claim_summary_lines.append(
            f"- [{conf_label}] {c.text[:100]}"
        )
    claim_summary = "\n".join(claim_summary_lines)

    prompt = f"""你是一个报告质量审核员。检查以下研究报告的质量。

【研究问题】
{chr(10).join(f'- {q}' for q in state.questions)}

【已验证的结论（含可信度）】
{claim_summary}

【报告开头 1200 字】
{state.report[:1200]}

请检查：
1. 报告中是否有超过 100 字连续段落完全没有 [来源X] 引用？（无依据推测）
2. 报告结构是否完整（有引言、正文分析、结论段落）
3. 是否有看起来像 LLM 幻觉的内容（听起来合理但缺乏证据支持）
4. 所有研究问题是否都被回答了
5. 低可信度结论是否被正确标注或谨慎处理

输出格式：
PASSED: 是/否
问题列表（每行一个，如果没有问题则留空）：
- 问题1
- 问题2
修改建议（如果需要改写，给出具体方向）：
建议1
建议2"""
    try:
        result = llm.generate(prompt)
        has_issues = not result.strip().upper().startswith("PASSED: 是")

        if has_issues:
            in_suggestions = False
            for line in result.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("修改建议"):
                    in_suggestions = True
                    continue
                if line.startswith("- ") and not in_suggestions:
                    text = line[2:]
                    if len(text) > 3:
                        issues.append(text)
                elif in_suggestions and line and "建议" not in line[:4]:
                    suggestions.append(line)

    except Exception:
        pass

    return issues, suggestions
