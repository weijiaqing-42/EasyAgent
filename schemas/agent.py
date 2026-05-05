from pydantic import BaseModel
from typing import List, Optional


class AgentCreate(BaseModel):
    agent_id: str                      # 用户自定义ID，如 "sql_helper"
    name: str                          # 显示名称
    description: str                   # Agent描述/角色定义（作为system prompt）
    tools: List[str] = []              # 工具列表，如 ["calculator", "search"]


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    description: str
    tools: List[str]


class AgentChatRequest(BaseModel):
    agent_id: str
    message: str
    conversation_history: List[dict] = []   # [{"role":"user","content":"..."}]
    kb_id: Optional[int] = None