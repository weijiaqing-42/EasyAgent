from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class ConversationCreate(BaseModel):
    title: str = "新对话"
    agent_id: Optional[str] = None


class ConversationResponse(BaseModel):
    id: int
    title: str
    agent_id: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    content: str
    kb_id: Optional[int] = None    # 是否挂载知识库


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True