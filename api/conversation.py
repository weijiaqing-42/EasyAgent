from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models.user import User
from models.conversation import Conversation, Message
from schemas.conversation import (
    ConversationCreate, ConversationResponse,
    MessageCreate, MessageResponse,
)
from utils.security import get_current_user
from core.agent_engine import run_agent
from core.rag_pipeline import rag_generate
from models.knowledge_base import KnowledgeBase
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import settings

router = APIRouter(prefix="/api/conversations", tags=["会话管理"])


def _get_llm_reply(messages: List[Message], new_content: str) -> str:
    """普通多轮对话"""
    llm = ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
        temperature=0.7,
        # ✅ 关闭思考模式，返回纯文本
        model_kwargs={"extra_body": {"enable_thinking": False}},
    )
    history = [SystemMessage(content="你是一个智能助手，请用中文回答。")]
    for msg in messages[-10:]:
        if msg.role == "user":
            history.append(HumanMessage(content=msg.content))
        else:
            history.append(AIMessage(content=msg.content))
    history.append(HumanMessage(content=new_content))
    resp = llm.invoke(history)
    return resp.content


@router.post("/", response_model=ConversationResponse, summary="创建新会话")
def create_conversation(
    conv_in: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = Conversation(
        user_id=current_user.id,
        title=conv_in.title,
        agent_id=conv_in.agent_id,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/", response_model=List[ConversationResponse], summary="获取会话列表")
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()


@router.delete("/{conv_id}", summary="删除会话")
def delete_conversation(
    conv_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.query(Message).filter(Message.conversation_id == conv_id).delete()
    db.delete(conv)
    db.commit()
    return {"message": "删除成功"}


@router.get("/{conv_id}/messages", response_model=List[MessageResponse], summary="获取历史消息")
def get_messages(
    conv_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return db.query(Message).filter(
        Message.conversation_id == conv_id
    ).order_by(Message.created_at).all()


@router.post("/{conv_id}/chat", response_model=MessageResponse, summary="发送消息并获取回复")
def chat(
    conv_id: int,
    msg_in: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    user_msg = Message(conversation_id=conv_id, role="user", content=msg_in.content)
    db.add(user_msg)
    db.commit()

    history_msgs = db.query(Message).filter(
        Message.conversation_id == conv_id
    ).order_by(Message.created_at).all()

    try:
        if conv.agent_id:
            history_dicts = [{"role": m.role, "content": m.content} for m in history_msgs[:-1]]
            reply_content = run_agent(conv.agent_id, msg_in.content, history_dicts)
        elif msg_in.kb_id:
            kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == msg_in.kb_id).first()
            if not kb:
                raise HTTPException(status_code=404, detail="知识库不存在")
            reply_content = rag_generate(kb.collection_name, msg_in.content)
        else:
            reply_content = _get_llm_reply(history_msgs[:-1], msg_in.content)
    except Exception as e:
        reply_content = f"处理请求时发生错误：{str(e)}"

    assistant_msg = Message(conversation_id=conv_id, role="assistant", content=reply_content)
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
    return assistant_msg