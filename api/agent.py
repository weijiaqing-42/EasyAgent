from fastapi import APIRouter, HTTPException
from typing import List
from schemas.agent import AgentCreate, AgentResponse, AgentChatRequest
from core.agent_engine import (
    register_agent, list_agents, get_agent_info,
    delete_agent, run_agent, AVAILABLE_TOOLS,
)

router = APIRouter(prefix="/api/agents", tags=["Agent管理"])


@router.get("/tools", summary="获取可用工具列表")
def get_available_tools():
    return {"tools": list(AVAILABLE_TOOLS.keys())}


@router.post("/", response_model=AgentResponse, summary="创建自定义Agent")
def create_agent(agent_in: AgentCreate):
    # 校验工具合法性
    for t in agent_in.tools:
        if t not in AVAILABLE_TOOLS:
            raise HTTPException(status_code=400, detail=f"工具 '{t}' 不存在")
    result = register_agent(
        agent_id=agent_in.agent_id,
        name=agent_in.name,
        description=agent_in.description,
        tools=agent_in.tools,
    )
    return result


@router.get("/", response_model=List[AgentResponse], summary="获取Agent列表")
def get_agents():
    return list_agents()


@router.get("/{agent_id}", response_model=AgentResponse, summary="获取Agent详情")
def get_agent(agent_id: str):
    agent = get_agent_info(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent不存在")
    return agent


@router.delete("/{agent_id}", summary="删除Agent")
def remove_agent(agent_id: str):
    if not delete_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent不存在")
    return {"message": "删除成功"}


@router.post("/chat", summary="直接与Agent对话（不绑定会话）")
def agent_chat(req: AgentChatRequest):
    try:
        reply = run_agent(req.agent_id, req.message, req.conversation_history)
        return {"reply": reply}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent执行错误: {str(e)}")