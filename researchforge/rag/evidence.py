"""
EvidenceRetriever — 在当前任务的已抓取文档中检索证据片段

替代旧的 RAGPipeline（它检索"历史研究"）。
改为在当前任务的 documents 中检索与问题相关的段落。
"""

from typing import List
from .retriever import Document


class EvidenceRetriever:
    """证据检索器：在已抓取的文档中找与问题相关的段落"""

    def __init__(self):
        self.documents: List[Document] = []

    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)

    def retrieve(self, question: str, top_k: int = 5) -> List[dict]:
        """
        检索相关证据片段

        返回: [{"source_id": str, "text": str, "score": float}]
        """
        results = []
        # 用中英文分词：按空格和标点拆分
        import re as _re
        # 匹配英文/数字词（中文暂不拆分，直接用子串匹配）
        _word_pat = _re.compile(r'[a-z0-9]+', _re.UNICODE)
        q_words = _word_pat.findall(question.lower())
        q_english = set(q_words)
        # 同时用中文子串匹配
        question_lower = question.lower()
        for doc in self.documents:
            if not doc.content:
                continue
            # 按段落拆分
            paragraphs = doc.content.replace("\r\n", "\n").split("\n\n")
            for para in paragraphs:
                if len(para) < 20:
                    continue
                p_words = set(_word_pat.findall(para.lower()))
                # 中英文都匹配：英文精确匹配 + 中文子串匹配
                matched = bool(q_english & p_words)
                if not matched:
                    # 中文子串匹配：问题中的中文是否出现在段落里
                    for c in question_lower:
                        if ord(c) > 127 and c in para.lower():
                            matched = True
                            break
                if matched:
                    # 计算分数（英文词重叠 + 中文子串覆盖）
                    en_overlap = len(q_english & p_words)
                    cn_hits = sum(1 for c in question_lower if ord(c) > 127 and c in para.lower())
                    cn_total = sum(1 for c in question_lower if ord(c) > 127)
                    score = (en_overlap + cn_hits / max(cn_total, 1)) / max(len(q_english) + 1, 1)
                    results.append({
                        "source_id": doc.source_id,
                        "text": para[:1000],
                        "score": score,
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
