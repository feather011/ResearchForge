"""
ResearchForge 配置管理

用法：
    from researchforge.service.config import settings
    print(settings.API_KEY)
    print(settings.MODEL)

环境变量加载优先级：
1. 系统环境变量（最高）
2. .env 文件
3. 默认值（最低）
"""

import os
import json
from pathlib import Path
from typing import Optional


def _find_project_root() -> Path:
    """从当前文件向上找项目根目录（包含 researchforge 目录）"""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "researchforge").is_dir():
            return parent
    return Path.cwd()


def _load_dotenv(dotenv_path: Path) -> dict:
    """加载 .env 文件（简单实现，不依赖第三方库）"""
    result = {}
    if not dotenv_path.exists():
        return result

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # 去掉引号
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            result[key] = value

    return result


PROJECT_ROOT = _find_project_root()
ENV_FILE = PROJECT_ROOT / ".env"


class Settings:
    """应用配置"""

    def __init__(self):
        # 加载 .env 文件
        dotenv_vars = _load_dotenv(ENV_FILE)

        # ========== API 配置 ==========
        self.LLM_PROVIDER: str = (
            os.environ.get("LLM_PROVIDER")
            or dotenv_vars.get("LLM_PROVIDER")
            or "bailian"
        )
        self.MODEL: str = (
            os.environ.get("RESEARCHFORGE_MODEL")
            or dotenv_vars.get("RESEARCHFORGE_MODEL")
            or "kimi-k2.6"
        )
        self.OLLAMA_BASE_URL: str = (
            os.environ.get("OLLAMA_BASE_URL")
            or dotenv_vars.get("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        self.API_KEY: str = (
            os.environ.get("DASHSCOPE_API_KEY")
            or dotenv_vars.get("DASHSCOPE_API_KEY")
            or ""
        )
        self.API_BASE: str = (
            os.environ.get("DASHSCOPE_BASE_URL")
            or dotenv_vars.get("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.LLM_TIMEOUT: int = int(
            os.environ.get("LLM_TIMEOUT")
            or dotenv_vars.get("LLM_TIMEOUT")
            or 300
        )

        # ========== 服务配置 ==========
        self.HOST: str = (
            os.environ.get("HOST")
            or dotenv_vars.get("HOST")
            or "0.0.0.0"
        )
        self.PORT: int = int(
            os.environ.get("PORT")
            or dotenv_vars.get("PORT")
            or 8002
        )
        self.DEBUG: bool = (
            os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
            or dotenv_vars.get("DEBUG", "").lower() in ("1", "true", "yes")
        )

        # ========== 限流配置 ==========
        self.RATE_LIMIT_MAX: int = int(
            os.environ.get("RATE_LIMIT_MAX")
            or dotenv_vars.get("RATE_LIMIT_MAX")
            or 20
        )
        self.RATE_LIMIT_WINDOW: int = int(
            os.environ.get("RATE_LIMIT_WINDOW")
            or dotenv_vars.get("RATE_LIMIT_WINDOW")
            or 60
        )

        # ========== 记忆系统 ==========
        self.MEMORY_BUFFER_MAX_TOKENS: int = int(
            os.environ.get("MEMORY_BUFFER_MAX_TOKENS")
            or dotenv_vars.get("MEMORY_BUFFER_MAX_TOKENS")
            or 4000
        )
        self.MEMORY_SUMMARY_MAX_TOKENS: int = int(
            os.environ.get("MEMORY_SUMMARY_MAX_TOKENS")
            or dotenv_vars.get("MEMORY_SUMMARY_MAX_TOKENS")
            or 2000
        )
        self.MEMORY_RAG_TOP_K: int = int(
            os.environ.get("MEMORY_RAG_TOP_K")
            or dotenv_vars.get("MEMORY_RAG_TOP_K")
            or 3
        )

        # ========== 研究配置 ==========
        self.PLANNER_MAX_STEPS: int = int(
            os.environ.get("PLANNER_MAX_STEPS")
            or dotenv_vars.get("PLANNER_MAX_STEPS")
            or 5
        )
        self.GROUPCHAT_MAX_ROUNDS: int = int(
            os.environ.get("GROUPCHAT_MAX_ROUNDS")
            or dotenv_vars.get("GROUPCHAT_MAX_ROUNDS")
            or 6
        )
        self.REACT_MAX_STEPS: int = int(
            os.environ.get("REACT_MAX_STEPS")
            or dotenv_vars.get("REACT_MAX_STEPS")
            or 5
        )
        self.REACT_MAX_REPLANS: int = int(
            os.environ.get("REACT_MAX_REPLANS")
            or dotenv_vars.get("REACT_MAX_REPLANS")
            or 2
        )

    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  API_KEY={'*' * 8 + self.API_KEY[-4:] if self.API_KEY else '<empty>'},\n"
            f"  MODEL={self.MODEL!r},\n"
            f"  API_BASE={self.API_BASE!r},\n"
            f"  HOST={self.HOST!r}, PORT={self.PORT}, DEBUG={self.DEBUG},\n"
            f"  LLM_TIMEOUT={self.LLM_TIMEOUT},\n"
            f"  RATE_LIMIT_MAX={self.RATE_LIMIT_MAX},\n"
            f"  PLANNER_MAX_STEPS={self.PLANNER_MAX_STEPS},\n"
            f")"
        )


# 全局单例
settings = Settings()
