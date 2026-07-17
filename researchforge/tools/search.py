import urllib.request
import urllib.parse
import json
import re
import logging

from researchforge.core.react_engine import BaseTool

logger = logging.getLogger("SearchTool")


class WebSearchTool(BaseTool):
    """网页搜索工具（Bing 搜索 + LLM 回退）"""

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

    def _search_google_first(self, query: str) -> str:
        """Google 优先，Bing 兜底，DuckDuckGo 最后"""
        try:
            from bs4 import BeautifulSoup
            import requests
        except ImportError:
            return self._search_duckduckgo(query)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        # 1. Google
        try:
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&hl=zh-CN"
            resp = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(resp.text, "html.parser")
            parts = []
            for res in soup.select("div.g")[:5]:
                title_el = res.select_one("h3")
                snippet_el = res.select_one("div[data-sncf], span.aCOpRe, div.VwiC3b, .st")
                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                if title:
                    parts.append(f"[{title}] {snippet[:200]}")
            if parts:
                return "\n\n".join(parts)
        except Exception:
            pass

        # 2. Bing V2
        try:
            url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
            resp = requests.get(url, headers=headers, proxies={"http": "", "https": ""}, timeout=8)
            soup = BeautifulSoup(resp.text, "html.parser")
            parts = []
            for res in soup.select("li.b_algo, .b_algo, .b_algoSquare")[:5]:
                title_el = res.select_one("h2 a")
                snippet_el = res.select_one(".b_caption p, .b_lineclamp2, .b_vList .b_caption p")
                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                if title:
                    parts.append(f"[{title}] {snippet[:200]}")
            if parts:
                return "\n\n".join(parts)
        except Exception:
            pass

        # 3. DuckDuckGo
        return self._search_duckduckgo(query)

    def _search_duckduckgo(self, query: str) -> str:
        """备用：DuckDuckGo API"""
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchForge/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
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
            response = urllib.request.urlopen(url, timeout=5)
            html = response.read().decode('utf-8')
            import re
            text = re.sub(r'<[^>]+>', '', html)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:1000] + "..." if len(text) > 1000 else text
        except Exception as e:
            return f"错误: 抓取网页失败 - {e}"
