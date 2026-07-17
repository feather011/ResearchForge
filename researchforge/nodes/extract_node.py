"""
ExtractNode — 证据提取节点

从抓取的文档中提取与问题相关的证据片段
"""

from typing import List
from ..orchestration.research_state import Evidence, Document
from ..rag.evidence import EvidenceRetriever


def run_extract_node(
    documents: List[Document],
    questions: List[str],
) -> List[Evidence]:
    """
    从文档中提取证据片段

    原理：
    1. 把文档注入 EvidenceRetriever
    2. 对每个问题检索相关段落
    3. 段落按词重叠度排序

    返回: [Evidence(id, source_id, text), ...]
    """
    retriever = EvidenceRetriever()
    retriever.add_documents(documents)

    all_evidences: List[Evidence] = []
    seen_texts = set()

    for q in questions:
        results = retriever.retrieve(q, top_k=5)
        for r in results:
            key = r["text"][:100]
            if key in seen_texts:
                continue
            seen_texts.add(key)
            all_evidences.append(Evidence(
                id=f"ev_{len(all_evidences)}",
                source_id=r["source_id"],
                text=r["text"],
            ))

    return all_evidences
