from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from database import create_all_tables
from api.user import router as user_router
from api.conversation import router as conv_router
from api.agent import router as agent_router
from api.knowledge_base import router as kb_router
from config import settings

# ✅ 启动前验证关键配置
if not settings.openai_api_key or settings.openai_api_key.startswith("sk-xxx"):
    raise RuntimeError(
        "\n\n❌ 未检测到有效的 API Key！\n"
        "请打开 .env 文件，将 OPENAI_API_KEY 替换为你的真实阿里云百炼 API Key\n"
        "获取地址：https://bailian.console.aliyun.com\n"
    )

create_all_tables()

app = FastAPI(
    title="多智能体对话与知识问答平台",
    description="基于FastAPI + LangChain + RAG + Milvus 构建的智能对话平台",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router)
app.include_router(conv_router)
app.include_router(agent_router)
app.include_router(kb_router)

# 挂载静态文件
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")


@app.get("/health", tags=["健康检查"])
def health_check():
    return {
        "status": "running",
        "model": settings.openai_model,
        "base_url": settings.openai_base_url,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)