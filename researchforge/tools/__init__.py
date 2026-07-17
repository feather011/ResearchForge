"""工具包 - 统一导出所有工具（仅搜索/抓取被主流程使用，其他工具保留供未来扩展）"""

from .search import WebSearchTool, WebScraperTool


# 保持向后兼容，供旧测试引用
def get_all_tools():
    return [WebSearchTool(), WebScraperTool()]
