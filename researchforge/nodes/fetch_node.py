"""
FetchNode — 网页抓取节点

对搜索到的 URL 抓取正文，转为 Document
（目前 DuckDuckGo 不返回 URL，使用搜索摘要作为 Document 内容）
"""

from typing import List
from ..orchestration.research_state import Source, Document


def run_fetch_node(
    sources: List[Source],
    max_pages: int = 3,
) -> List[Document]:
    """
    抓取/整理来源正文

    原理：
    1. 对每个 Source 尝试抓取 URL（如果有 URL 且可访问）
    2. 没有 URL 的 Source 直接用 snippet 作为文档内容
    3. 返回 Document 列表供 EvidenceRetriever 检索

    返回: [Document(source_id, content, url, title), ...]
    """
    from ..tools.search import WebScraperTool
    scraper = WebScraperTool()
    documents: List[Document] = []

    for i, src in enumerate(sources[:max_pages]):
        content = ""

        # 有 URL 就尝试抓取
        if src.url:
            content = scraper.run(url=src.url)
            if content.startswith("错误"):
                content = ""  # 抓取失败，fallback

        # 没有 URL 或抓取失败，用 snippet
        if not content:
            content = src.snippet

        documents.append(Document(
            source_id=src.id,
            content=content[:2000],
            url=src.url,
            title=src.title,
        ))

    return documents
