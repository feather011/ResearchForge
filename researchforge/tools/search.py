import urllib.request
import urllib.parse
import json
import re
import logging

from researchforge.core.react_engine import BaseTool

logger = logging.getLogger("SearchTool")


class WebSearchTool(BaseTool):
    """网页搜索工具（Google → Bing 级联，带结果相关性过滤）"""

    def __init__(self, use_real_search: bool = True):
        self.use_real_search = use_real_search

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "搜索网页信息，返回相关结果"

    def run(self, query: str, **kwargs) -> str:
        if not self.use_real_search:
            return f"[模拟搜索] 关于「{query}」的结果"

        try:
            return self._search_google_first(query)
        except Exception as e:
            logger.warning(f"搜索失败: {e}")
            return f"[搜索失败: {e}]"

    def _extract_keywords(self, query: str) -> set:
        """从查询中提取有效关键词"""
        return set(w for w in query.lower().split() if len(w) > 1)

    def _results_are_relevant(self, parts: list, keywords: set) -> bool:
        """检查结果集是否相关：至少 1 条结果的标题命中关键词"""
        if not parts or not keywords:
            return False
        for p in parts:
            bracket_end = p.find("]")
            title = p[1:bracket_end].lower() if bracket_end > 0 else p.lower()
            if any(kw in title for kw in keywords):
                return True
        return False

    def _scrape_google(self, query: str, headers: dict) -> list:
        """抓取 Google 搜索结果"""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        try:
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&hl=zh-CN"
            resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            parts = []
            for res in soup.select("div.g")[:5]:
                title_el = res.select_one("h3")
                snippet_el = res.select_one("div[data-sncf], span.aCOpRe, div.VwiC3b, .st")
                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                if title:
                    parts.append(f"[{title}] {snippet[:200]}")
            return parts
        except Exception:
            return []

    def _scrape_bing(self, query: str, headers: dict) -> list:
        """抓取 Bing 搜索结果"""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        try:
            url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
            resp = requests.get(url, headers=headers, proxies={"http": "", "https": ""}, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            parts = []
            for res in soup.select("li.b_algo")[:5]:
                title_el = res.select_one("h2")
                snippet_el = res.select_one(".b_caption p")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                parts.append(f"[{title}] {snippet[:200]}")
            return parts
        except Exception:
            return []

    def _search_duckduckgo(self, query: str) -> str:
        """备用：DuckDuckGo API"""
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchForge/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        parts = []
        if data.get("AbstractText"):
            parts.append(f"[摘要] {data['AbstractText']}")
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
        return "\n\n".join(parts) if parts else ""

    def _search_google_first(self, query: str) -> str:
        """Google 优先，Bing 兜底，都失败则降级到单关键词重试"""
        try:
            from bs4 import BeautifulSoup
            import requests
        except ImportError:
            return self._search_duckduckgo(query) or f"未找到「{query}」的相关结果"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        keywords = self._extract_keywords(query)

        # 尝试原始查询 → Google
        parts = self._scrape_google(query, headers)
        if self._results_are_relevant(parts, keywords):
            return "\n\n".join(parts)

        # 尝试原始查询 → Bing
        parts = self._scrape_bing(query, headers)
        if self._results_are_relevant(parts, keywords):
            return "\n\n".join(parts)

        # 降级：取查询中第一个长词作为关键词重试（解决"文艺复兴 起源"→Google/Bing 分词为"文艺"的问题）
        words = [w for w in query.split() if len(w) > 1]
        if words:
            first_word = words[0]
            # 用第一个显著词尝试 Google
            parts = self._scrape_google(first_word, headers)
            if parts:
                return "\n\n".join(parts)
            # 再试 Bing
            parts = self._scrape_bing(first_word, headers)
            if parts:
                return "\n\n".join(parts)

        # DuckDuckGo 最后尝试
        ddg_result = self._search_duckduckgo(query)
        if ddg_result:
            return ddg_result

        return f"未找到「{query}」的相关结果"

    def _search_duckduckgo(self, query: str) -> str:
        """备用：DuckDuckGo API"""
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchForge/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        parts = []
        if data.get("AbstractText"):
            parts.append(f"[摘要] {data['AbstractText']}")
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
        return "\n\n".join(parts) if parts else f"未找到「{query}」的相关结果"


class WebScraperTool(BaseTool):
    """网页抓取工具"""

    @property
    def name(self) -> str:
        return "web_scraper"

    @property
    def description(self) -> str:
        return "抓取指定网页的内容"

    def run(self, url: str, **kwargs) -> str:
        try:
            import urllib.request
            response = urllib.request.urlopen(url, timeout=2)
            html = response.read().decode('utf-8')
            import re
            text = re.sub(r'<[^>]+>', '', html)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:1000] + "..." if len(text) > 1000 else text
        except Exception as e:
            return f"错误: 抓取网页失败 - {e}"
