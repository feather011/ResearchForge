"""
FetchNode — 网页抓取节点

对搜索到的 URL 并行抓取正文，转为 Document
超时 2s，不可达的站快速跳过
"""

from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..orchestration.research_state import Source, Document


def _fetch_single(src: Source) -> Document:
    """抓取单个来源"""
    from ..tools.search import WebScraperTool
    scraper = WebScraperTool()
    content = ""

    if src.url:
        content = scraper.run(url=src.url)

    if not content:
        content = src.snippet

    return Document(
        source_id=src.id,
        content=content[:2000],
        url=src.url,
        title=src.title,
    )


def run_fetch_node(
    sources: List[Source],
    max_pages: int = 3,
) -> List[Document]:
    """
    并行抓取来源正文

    用 ThreadPoolExecutor 并行抓取，避免串行等超时。
    每个来源最多等 2s，不可达快速跳过。

    返回: [Document(source_id, content, url, title), ...]
    """
    documents: List[Document] = []

    with ThreadPoolExecutor(max_workers=min(len(sources), max_pages, 5)) as pool:
        futures = {pool.submit(_fetch_single, src): src for src in sources[:max_pages]}
        for f in as_completed(futures):
            try:
                documents.append(f.result())
            except Exception:
                pass

    return documents
