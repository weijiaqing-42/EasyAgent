"""
Agent引擎：基于LangChain构建自定义Agent，支持多工具链式调用
"""
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from config import settings

# ===================== 内置工具定义 =====================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入：数学表达式字符串，如 '2 + 3 * 4'"""
    try:
        allowed = set("0123456789+-*/().,% ")
        if not all(c in allowed for c in expression):
            return "表达式包含非法字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def text_word_count(text: str) -> str:
    """统计文本的字数（中文字符数+英文单词数）"""
    import re
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    return f"中文字符：{chinese_chars} 个，英文单词：{english_words} 个，合计约 {chinese_chars + english_words} 词"


AVAILABLE_TOOLS: Dict[str, any] = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "text_word_count": text_word_count,
}

_agent_registry: Dict[str, dict] = {}


def register_agent(agent_id: str, name: str, description: str, tools: List[str]) -> dict:
    _agent_registry[agent_id] = {
        "agent_id": agent_id,
        "name": name,
        "description": description,
        "tools": tools,
    }
    return _agent_registry[agent_id]


def list_agents() -> List[dict]:
    return list(_agent_registry.values())


def get_agent_info(agent_id: str) -> Optional[dict]:
    return _agent_registry.get(agent_id)


def delete_agent(agent_id: str) -> bool:
    if agent_id in _agent_registry:
        del _agent_registry[agent_id]
        return True
    return False


def build_agent_executor(system_prompt: str, tool_names: List[str]) -> AgentExecutor:
    """构建AgentExecutor实例"""
    selected_tools = [
        AVAILABLE_TOOLS[t] for t in tool_names if t in AVAILABLE_TOOLS
    ]

    llm = ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
        temperature=0,
        # ✅ 关闭 qwen3.6-plus 的思考模式，保证 Tool Call 格式正确
        model_kwargs={"extra_body": {"enable_thinking": False}},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, selected_tools, prompt)
    return AgentExecutor(agent=agent, tools=selected_tools, verbose=True, max_iterations=5)


def run_agent(
    agent_id: str,
    message: str,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    agent_info = _agent_registry.get(agent_id)
    if not agent_info:
        raise ValueError(f"Agent '{agent_id}' 不存在，请先创建")

    executor = build_agent_executor(
        system_prompt=agent_info["description"],
        tool_names=agent_info["tools"],
    )

    history = []
    for msg in (conversation_history or []):
        if msg.get("role") == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            history.append(AIMessage(content=msg["content"]))

    result = executor.invoke({
        "input": message,
        "chat_history": history,
    })
    return result.get("output", "Agent未返回结果")