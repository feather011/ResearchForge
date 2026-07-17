"""
Pydantic 请求/响应模型
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


# ==================== 请求模型 ====================

class ResearchRequest(BaseModel):
    """启动研究请求"""
    topic: str = Field(..., description="研究主题", min_length=1, max_length=500)
    mode: str = Field("fast", description="运行模式: fast / standard / deep")


class ReviewRequest(BaseModel):
    """人工审核请求"""
    response: str = Field(..., description="通过 / 驳回 / 修改意见")
    comments: Optional[str] = Field(None, description="意见")


# ==================== 响应模型 ====================

class ResearchResponse(BaseModel):
    """启动研究响应"""
    research_id: str
    topic: str
    state: str
    message: str


class StatusResponse(BaseModel):
    """状态查询响应"""
    research_id: Optional[str] = None
    state: str
    topic: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    results_count: int = 0
    pending_interventions: List[Dict] = []
    checkpoints_count: int = 0


class CheckpointResponse(BaseModel):
    """检查点列表响应"""
    checkpoints: List[Dict[str, Any]]


class ReviewResponse(BaseModel):
    """审核响应"""
    state: str
    message: str
    report: Optional[str] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None
