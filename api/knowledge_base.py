import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models.user import User
from models.knowledge_base import KnowledgeBase, Document
from schemas.knowledge_base import KBCreate, KBResponse, DocumentResponse
from utils.security import get_current_user
from core.document_parser import load_and_split
from core.rag_pipeline import add_documents_to_milvus

router = APIRouter(prefix="/api/knowledge-bases", tags=["知识库管理"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=KBResponse, summary="创建知识库")
def create_kb(
    kb_in: KBCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    import uuid
    collection_name = f"kb_{current_user.id}_{uuid.uuid4().hex[:8]}"
    kb = KnowledgeBase(
        user_id=current_user.id,
        name=kb_in.name,
        description=kb_in.description,
        collection_name=collection_name,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


@router.get("/", response_model=List[KBResponse], summary="获取知识库列表")
def list_kbs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(KnowledgeBase).filter(
        KnowledgeBase.user_id == current_user.id
    ).all()


@router.delete("/{kb_id}", summary="删除知识库")
def delete_kb(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id,
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    db.query(Document).filter(Document.kb_id == kb_id).delete()
    db.delete(kb)
    db.commit()
    return {"message": "删除成功"}


@router.post("/{kb_id}/upload", response_model=DocumentResponse, summary="上传文档")
async def upload_document(
    kb_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id,
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    allowed_ext = {".pdf", ".docx", ".doc", ".md", ".markdown"}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    save_path = os.path.join(UPLOAD_DIR, f"{kb_id}_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = load_and_split(save_path)
        count = add_documents_to_milvus(kb.collection_name, chunks)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    doc = Document(
        kb_id=kb_id,
        filename=file.filename,
        file_type=ext.lstrip(".").lower(),
        chunk_count=count,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    os.remove(save_path)
    return doc


@router.get("/{kb_id}/documents", response_model=List[DocumentResponse], summary="获取知识库文档列表")
def list_documents(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == current_user.id,
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return db.query(Document).filter(Document.kb_id == kb_id).all()