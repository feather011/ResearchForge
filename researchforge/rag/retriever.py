"""
Multi-Stage RAG - 多阶段检索增强生成

流程：
1. 召回阶段：从多个数据源检索相关文档
2. 融合阶段：合并多个检索结果
3. 精排阶段：重新排序，选择最相关的文档
4. 生成阶段：基于检索结果生成答案

组件：
- VectorRetriever：向量检索
- BM25Retriever：BM25检索
- FusionRetriever：融合检索
- Reranker：精排器
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
import math

logger = logging.getLogger("RAG")


@dataclass
class Document:
    """文档"""
    content: str            # 内容
    metadata: Dict = field(default_factory=dict)  # 元数据
    score: float = 0.0      # 相关性分数
    source: str = ""        # 来源


class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """检索文档"""
        pass


class VectorRetriever(BaseRetriever):
    """
    向量检索器

    使用向量相似度检索文档
    """

    def __init__(self):
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]):
        """添加文档"""
        self.documents.extend(documents)
        logger.info(f"添加 {len(documents)} 个文档到向量检索器")

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示（基于词频+结构的 4 维特征）"""
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
        content_hash = hash(text[:100]) % 10000
        return [
            float(word_count),
            float(unique_words),
            float(avg_word_len),
            float(content_hash),
        ]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度

        注意：这里使用简化的实现
        """
        if not vec1 or not vec2:
            return 0.0
        return 1.0 / (1.0 + abs(vec1[0] - vec2[0]))

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """向量检索"""
        if not self.documents:
            return []

        # 获取查询向量
        query_embedding = self._get_embedding(query)

        # 计算相似度
        results = []
        for doc in self.documents:
            doc_embedding = self._get_embedding(doc.content)
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            doc.score = similarity
            results.append(doc)

        # 排序并返回top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class BM25Retriever(BaseRetriever):
    """
    BM25检索器

    使用BM25算法检索文档
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.doc_count = 0
        self.avg_doc_length = 0
        self.doc_freqs: Dict[str, int] = {}  # 词频
        self.doc_lengths: List[int] = []  # 文档长度

    def add_documents(self, documents: List[Document]):
        """添加文档"""
        self.documents.extend(documents)
        self.doc_count = len(self.documents)

        # 计算文档长度
        self.doc_lengths = [len(doc.content) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

        # 计算词频
        for doc in self.documents:
            words = set(doc.content.split())
            for word in words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        logger.info(f"添加 {len(documents)} 个文档到BM25检索器")

    def _tokenize(self, text: str) -> List[str]:
        """分词（简化实现）"""
        return text.split()

    def _bm25_score(self, query_terms: List[str], doc_index: int) -> float:
        """计算BM25分数"""
        score = 0.0
        doc_length = self.doc_lengths[doc_index]

        for term in query_terms:
            if term not in self.doc_freqs:
                continue

            # 词频
            tf = self.documents[doc_index].content.count(term)

            # 逆文档频率
            df = self.doc_freqs[term]
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

            # BM25公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """BM25检索"""
        if not self.documents:
            return []

        # 分词
        query_terms = self._tokenize(query)

        # 计算BM25分数
        results = []
        for i, doc in enumerate(self.documents):
            score = self._bm25_score(query_terms, i)
            doc.score = score
            results.append(doc)

        # 排序并返回top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class FusionRetriever(BaseRetriever):
    """
    融合检索器

    融合多个检索器的结果
    """

    def __init__(self, retrievers: List[BaseRetriever], weights: List[float] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """融合检索"""
        # 从各个检索器获取结果
        all_results = []
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(query, top_k=top_k * 2)  # 获取更多结果用于融合
            for doc in results:
                doc.score *= weight
            all_results.extend(results)

        # 合并相同文档（基于内容）
        merged = {}
        for doc in all_results:
            if doc.content in merged:
                # 合并分数
                merged[doc.content].score += doc.score
            else:
                merged[doc.content] = doc

        # 排序并返回top_k
        results = list(merged.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class Reranker:
    """
    精排器

    重新排序文档
    """

    def __init__(self):
        pass

    def _calculate_relevance(self, query: str, document: str) -> float:
        """
        计算相关性分数

        注意：这里使用简化的实现
        实际项目中应该使用交叉编码器或重排序模型
        """
        # 简化实现：基于词重叠
        query_words = set(query.split())
        doc_words = set(document.split())
        overlap = len(query_words & doc_words)
        return overlap / len(query_words) if query_words else 0.0

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """精排"""
        # 计算相关性分数
        for doc in documents:
            relevance = self._calculate_relevance(query, doc.content)
            doc.score = doc.score * 0.3 + relevance * 0.7  # 混合原始分数和相关性

        # 排序并返回top_k
        documents.sort(key=lambda x: x.score, reverse=True)
        return documents[:top_k]
