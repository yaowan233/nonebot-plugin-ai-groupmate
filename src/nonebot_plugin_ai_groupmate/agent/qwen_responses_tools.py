from typing import Any

from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage


def _response_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def create_qwen_code_interpreter_tool(model: Any) -> BaseTool:
    """Wrap DashScope code_interpreter in a separate Responses request.

    DashScope documents code_interpreter as incompatible with Function Calling.
    The main bot Agent depends on local function tools, so the sandbox request is
    isolated behind one local tool instead of mixing both tool families in a
    single Responses request.
    """
    code_interpreter_model = model.bind_tools(
        [{"type": "code_interpreter"}]
    )

    @tool("qwen_code_interpreter")
    async def qwen_code_interpreter(task: str) -> str:
        """Use Qwen's official Python sandbox for exact calculation or data analysis.

        Pass a self-contained task with all required numbers and constraints. The
        returned text is analysis material; use reply_user to send the final answer.
        """
        response = await code_interpreter_model.ainvoke([HumanMessage(content=task)])
        result = _response_text(response)
        if not result:
            raise RuntimeError("千问代码解释器没有返回可读文本")
        return result

    return qwen_code_interpreter
