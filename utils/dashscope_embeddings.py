"""
阿里云百炼 text-embedding-v4 自定义Embeddings
解决单批次最大10条限制问题
"""
from typing import List
from langchain_openai import OpenAIEmbeddings
from config import settings


class DashScopeEmbeddings(OpenAIEmbeddings):
    """
    继承 OpenAIEmbeddings，重写批处理逻辑。
    text-embedding-v4 单批次上限为 10 条，自动分批。
    """

    # 强制批大小不超过10
    chunk_size: int = 10

    def __init__(self, **kwargs):
        super().__init__(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
            # 指定向量维度（text-embedding-v4 支持自定义维度）
            dimensions=settings.embedding_dimensions,
            # 每批最多10条，防止超出百炼限制
            chunk_size=10,
            **kwargs,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """分批嵌入文档列表（每批最多10条）"""
        all_embeddings = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch_embeddings = super().embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询文本"""
        return super().embed_query(text)