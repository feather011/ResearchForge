"""
SearchNode — 搜索节点

对每个子问题执行级联搜索（Google→Bing→DuckDuckGo），返回 Source 列表
直接用问题原文搜索，不调 LLM 提取关键词（慢模型上每次 15-25s，纯浪费）
"""

from typing import List
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
    并行搜索每个子问题

    直接用问题原文作为搜索词，并行搜索所有问题。
    不调 LLM 提取关键词——对 k2.6 这类慢模型，每次关键词提取 15-25s，4 个问题浪费 1-2 分钟。

    返回: [Source(id, url, title, snippet), ...]
    """
    all_sources: List[Source] = []
    seen_snippets = set()

    # 并行搜索
    with ThreadPoolExecutor(max_workers=min(len(questions), 5)) as pool:
        futures = {}
        for qi, question in enumerate(questions):
            futures[pool.submit(_search_single, question, sources_per_question, use_real_search)] = qi

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
) -> List[Source]:
    """单个问题搜索（并行 worker 用）"""
    tool = WebSearchTool(use_real_search=use_real_search)
    results: List[Source] = []

    # 直接用问题原文搜索，不调 LLM 提取关键词（k2.6 慢模型上每次 15-25s，纯浪费）
    keywords = question[:60]

    result = tool.run(keywords, timeout=5)

    # 搜索失败，直接返回占位
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
