from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class KBCreate(BaseModel):
    name: str
    description: Optional[str] = None


class KBResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    collection_name: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    chunk_count: int
    created_at: datetime

    class Config:
        from_attributes = True