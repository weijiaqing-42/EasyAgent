import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 强制加载 .env 文件（双保险）
load_dotenv(override=True)


class Settings(BaseSettings):
    # 阿里云百炼
    openai_api_key: str = ""
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_database: str = "multi_agent_db"

    # Milvus
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530

    # JWT
    secret_key: str = "change-this-secret"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def milvus_connection_args(self) -> dict:
        return {
            "host": self.milvus_host,
            "port": self.milvus_port,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
os.environ["OPENAI_API_KEY"] = settings.openai_api_key
os.environ["OPENAI_BASE_URL"] = settings.openai_base_url
# ✅ 启动时打印验证（Key只显示前8位，保护安全）
print(f"[Config] API Key 前8位: {settings.openai_api_key[:8] if settings.openai_api_key else '❌ 未设置'}")
print(f"[Config] Base URL: {settings.openai_base_url}")
print(f"[Config] 模型: {settings.openai_model}")