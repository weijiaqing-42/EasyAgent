"""
RAG全链路：Embedding → Milvus存储 → 语义检索 → LLM生成
"""
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import settings
from utils.dashscope_embeddings import DashScopeEmbeddings


def get_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings()


def get_vectorstore(collection_name: str) -> Milvus:
    """获取指定collection的Milvus实例（连接Docker中的Milvus）"""
    return Milvus(
        embedding_function=get_embeddings(),
        collection_name=collection_name,
        # ✅ 使用 host+port 格式，不使用本地文件URI
        connection_args=settings.milvus_connection_args,
    )


def add_documents_to_milvus(collection_name: str, documents: List[Document]) -> int:
    """将文档chunks写入Milvus"""
    embeddings = get_embeddings()
    Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args=settings.milvus_connection_args,
    )
    return len(documents)


def semantic_search(collection_name: str, query: str, top_k: int = 4) -> List[Document]:
    """语义检索"""
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.similarity_search(query, k=top_k)


def rag_generate(
    collection_name: str,
    question: str,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """RAG生成：检索 + LLM生成"""
    vectorstore = get_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
        temperature=0.3,
        model_kwargs={"extra_body": {"enable_thinking": False}},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个专业的知识问答助手。请根据以下检索到的上下文内容来回答用户问题。"
         "如果上下文中没有相关信息，请直接说明你不知道，不要编造答案。\n\n"
         "上下文：\n{context}"),
        ("human", "{question}"),
    ])

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)