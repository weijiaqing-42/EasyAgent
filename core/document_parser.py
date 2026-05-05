"""
文档解析模块：支持 PDF / Word / Markdown
"""
import os
import re
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_pdf(file_path: str) -> str:
    """解析 PDF 文件"""
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def parse_docx(file_path: str) -> str:
    """解析 Word 文件"""
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def parse_markdown(file_path: str) -> str:
    """解析 Markdown 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 去除markdown语法符号，保留纯文本
    content = re.sub(r"#{1,6}\s", "", content)
    content = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", content)
    content = re.sub(r"`{1,3}.*?`{1,3}", "", content, flags=re.DOTALL)
    return content


def load_and_split(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    加载文档并分块
    返回 LangChain Document 列表
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        raw_text = parse_pdf(file_path)
        file_type = "pdf"
    elif ext in (".docx", ".doc"):
        raw_text = parse_docx(file_path)
        file_type = "docx"
    elif ext in (".md", ".markdown"):
        raw_text = parse_markdown(file_path)
        file_type = "md"
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.create_documents(
        texts=[raw_text],
        metadatas=[{"source": os.path.basename(file_path), "file_type": file_type}],
    )
    return chunks