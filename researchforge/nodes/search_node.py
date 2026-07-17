"""
SearchNode — 搜索节点

对每个子问题执行 Bing 搜索，返回 Source 列表
搜索前用 LLM 提取关键词，提高搜索命中率
"""

from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from ..orchestration.research_state import Source
from ..tools.search import WebSearchTool


# 线程安全的来源 ID 生成器
_source_counter = count(1)


def _reset_counter():
    """仅用于测试"""
    global _source_counter
    _source_counter = count(1)


def run_search_node(
    questions: List[str],
    sources_per_question: int = 3,
    use_real_search: bool = True,
    llm=None,
) -> List[Source]:
    """
    并行搜索每个子问题（搜索前用 LLM 提取关键词）

    原理：
    1. 对每个子问题先用 LLM 提取短关键词（3-5个词）
    2. 用关键词调 Bing 搜索
    3. 把搜索结果转为标准化的 Source 对象

    返回: [Source(id, url, title, snippet), ...]
    """
    all_sources: List[Source] = []
    seen_snippets = set()

    # 并行搜索
    with ThreadPoolExecutor(max_workers=min(len(questions), 5)) as pool:
        futures = {}
        for qi, question in enumerate(questions):
            futures[pool.submit(_search_single, question, sources_per_question, use_real_search, llm)] = qi

        for f in as_completed(futures):
            try:
                results = f.result()
                all_sources.extend(results)
            except Exception:
                pass

    # 全局去重 + 重新编号
    deduped = []
    sid = 1
    for src in all_sources:
        key = src.snippet[:50]
        if key not in seen_snippets:
            seen_snippets.add(key)
            src.id = f"来源{sid}"
            sid += 1
            deduped.append(src)

    return deduped


def _search_single(
    question: str,
    sources_per_question: int,
    use_real_search: bool,
    llm=None,
) -> List[Source]:
    """单个问题搜索（并行 worker 用）"""
    tool = WebSearchTool(use_real_search=use_real_search)
    results: List[Source] = []

    # 提取搜索关键词（缩短长查询）
    if llm and len(question) > 20:
        try:
            kw_prompt = f"将以下查询简化为3-5个搜索关键词（中文或英文，空格分隔）: {question}"
            keywords = llm.generate(kw_prompt).strip()
            keywords = " ".join(keywords.split()[:5])
        except Exception:
            keywords = question
    else:
        keywords = question

    result = tool.run(keywords)

    # 搜索失败，英文兜底
    if result.startswith("[搜索失败") or result.startswith("[模拟搜索") or result == f"未找到「{keywords}」的相关结果":
        if not (keywords and all(ord(c) < 128 for c in keywords)):
            try:
                en_keywords = f"Renaissance {keywords}"
                result2 = tool.run(en_keywords)
                if result2 and not result2.startswith("["):
                    result = result2
            except Exception:
                pass
        if result.startswith("[搜索失败") or result.startswith("[模拟搜索") or result == f"未找到「{keywords}」的相关结果":
            results.append(Source(
                id=f"来源{next(_source_counter)}",
                title=question[:80],
                snippet=result[:300],
                url="",
            ))
            return results

    # 解析搜索结果
    blocks = result.split("\n\n")
    q_words = set(keywords.lower().split())
    for bi, block in enumerate(blocks[:sources_per_question]):
        if not block.strip():
            continue
        block_lower = block.lower()
        has_relevance = any(w in block_lower for w in q_words if len(w) > 1)
        if not has_relevance and bi == 0:
            has_relevance = True
        if not has_relevance:
            continue

        results.append(Source(
            id=f"来源{next(_source_counter)}",
            snippet=block[:300],
            title=block.split("\n")[0][:80] if "\n" in block else block[:80],
            url="",
        ))

    return results
