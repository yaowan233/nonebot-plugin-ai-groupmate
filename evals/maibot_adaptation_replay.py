"""Read-only live-material replay of expression learning and visual meme review.

Run in the bot container with --package-root pointing at a staged package.
Never starts NoneBot, changes the production expression bank, or sends messages.
Only the temporary output directory is written; DB transactions are read-only.
This checks actual helper functions with real models, not the full Agent graph.
"""

import os
import ast
import sys
import json
import types
import asyncio
import argparse
from typing import Any, Literal
from pathlib import Path
from datetime import datetime
from collections import Counter
from collections.abc import Sequence

from dotenv import dotenv_values
from loguru import logger
from pydantic import Field, BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.messages import HumanMessage, SystemMessage


def load_definitions(path: Path, names: set[str], namespace: dict[str, Any]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names), names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


async def replay(root: Path, output: Path, group: str) -> dict[str, Any]:
    # Avoid importing the plugin entrypoint and registering adapters/schedulers.
    for name, path in [("nonebot_plugin_ai_groupmate", root), ("nonebot_plugin_ai_groupmate.agent", root / "agent")]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    model_stub = types.ModuleType("nonebot_plugin_ai_groupmate.model")
    model_stub.ChatHistorySchema = types.SimpleNamespace  # type: ignore[attr-defined]
    sys.modules[model_stub.__name__] = model_stub

    from nonebot_plugin_ai_groupmate.agent import style_learning as learning
    from nonebot_plugin_ai_groupmate.config import ScopedConfig, create_chat_llm, create_vision_llm, create_chat_openai
    from nonebot_plugin_ai_groupmate.deepseek import DeepSeekChatOpenAI
    from nonebot_plugin_ai_groupmate.message_metadata import deduplicate_bot_echoes, parse_message_metadata
    from nonebot_plugin_ai_groupmate.agent.meme_visuals import build_meme_contact_sheet

    env = {key.lower(): value for key, value in dotenv_values("/app/.env").items() if value}
    env.update({key.lower(): value for key, value in os.environ.items()})
    values = {key.removeprefix("ai_groupmate__"): value for key, value in env.items() if key.startswith("ai_groupmate__")}
    engine = create_async_engine(env["sqlalchemy_database_url"])
    async with engine.connect() as connection:
        await connection.execute(text("SET TRANSACTION READ ONLY"))
        values.update((await connection.execute(text("SELECT overrides FROM nonebot_plugin_ai_groupmate_runtimeconfigoverride WHERE id=1"))).scalar_one_or_none() or {})
        history = (await connection.execute(text("""
            SELECT msg_id, session_id, user_id, user_name, content_type, content, created_at, media_id
            FROM nonebot_plugin_ai_groupmate_chathistory
            WHERE session_id=:group AND created_at > NOW() - INTERVAL '7 days'
            ORDER BY msg_id DESC LIMIT 1200
        """), {"group": group})).mappings().all()
        media = (await connection.execute(text("""
            SELECT m.media_id, m.file_path, m.description
            FROM nonebot_plugin_ai_groupmate_mediastorage m
            JOIN (SELECT media_id, max(msg_id) AS latest FROM nonebot_plugin_ai_groupmate_chathistory
                  WHERE session_id=:group AND media_id IS NOT NULL GROUP BY media_id) h ON h.media_id=m.media_id
            WHERE m.references >= 3 ORDER BY h.latest DESC LIMIT 16
        """), {"group": group})).mappings().all()
    await engine.dispose()
    cfg = ScopedConfig.model_validate(values)
    flash = create_chat_openai(cfg, "flash")
    visual = create_vision_llm(cfg) if cfg.vision_model else (create_chat_llm(cfg) if cfg.chat_multimodal else None)
    assert visual is not None, "No configured visual model"
    rows = [types.SimpleNamespace(**dict(row)) for row in reversed(history)]
    namespace: dict[str, Any] = {
        "Sequence": Sequence, "ChatHistorySchema": types.SimpleNamespace,
        "Counter": Counter, "re": __import__("re"),
        "deduplicate_bot_echoes": deduplicate_bot_echoes, "parse_message_metadata": parse_message_metadata,
    }
    load_definitions(root / "agent/chat_style.py", {"_human_expressions"}, namespace)
    sources = namespace["_human_expressions"](rows, group, (), bot_names=(cfg.bot_name,))[:60]
    assert len(sources) >= 12
    assert len({row.user_id for row, _ in sources}) >= 3
    output.mkdir(parents=True, exist_ok=True)
    bank_file = learning.bank_path(output / "expressions", group)
    await learning._learn(bank_file, group, sources, lambda: flash)
    bank = learning._read_bank(bank_file, group)
    assert bank.last_source_id, "Learning call failed or could not be parsed"
    scenarios = [
        "当前群友说：居然真让他整出来了，笑死；本轮想轻松接一句惊讶。",
        "当前群友认真求助：能解释一下原理吗？本轮要准确解释，不能用嘲讽和附和代替。",
        "当前群友说：你理解错了，我说的是图里右边那个。先承接纠正，再重新核对，不要装作已经确认。",
        "群友聊到零件报价贵得夸张，想跟着用短句吐槽价格。",
        "群友晒晚饭照片，大家看着很馋，想接一句表达羡慕。",
    ]
    selections = []
    for context in scenarios:
        chosen = await learning.contextual_style(root=output / "expressions", session_id=group, sources=[],
                                                recent_chat=context, model_factory=lambda: flash)
        selections.append({"input": context, "selected_context": chosen})

    image_root = Path("/app/data/nonebot_plugin_ai_groupmate/pics")
    media_by_id = {int(row["media_id"]): dict(row) for row in media}
    visible = [(mid, item["file_path"]) for mid, item in media_by_id.items() if (image_root / item["file_path"]).is_file()][:6]
    grid, ids = await asyncio.to_thread(build_meme_contact_sheet, image_root, visible)
    assert grid, "No readable candidate images"
    assert ids, "No readable candidate images"
    from base64 import b64decode
    (output / "candidate-grid.png").write_bytes(b64decode(grid.split(",", 1)[1]))
    meme_namespace: dict[str, Any] = {
        "BaseModel": BaseModel, "Field": Field, "Any": Any, "Literal": Literal, "Sequence": Sequence,
        "ChatHistorySchema": types.SimpleNamespace, "MemeSearchType": Literal["context", "content", "random"],
        "DeepSeekChatOpenAI": DeepSeekChatOpenAI,
        "json": json, "asyncio": asyncio, "logger": logger, "HumanMessage": HumanMessage, "SystemMessage": SystemMessage,
        "MEME_CONTEXT_HISTORY_LIMIT": 8, "MEME_CONTEXT_RERANK_TIMEOUT_SECONDS": 20,
        "MEME_CONTENT_RELEVANCE_THRESHOLD": 0.6, "MEME_CONTEXT_RELEVANCE_THRESHOLD": 0.4,
    }
    load_definitions(root / "agent/meme_tools.py", {
        "MemeContextCandidateScore", "MemeContextReview", "_parse_meme_visual_review", "_history_content_body", "_format_meme_context", "_rerank_meme_candidates_for_context",
    }, meme_namespace)
    candidates = [(mid, item["description"]) for mid, item in media_by_id.items() if mid in ids]
    review_metadata: dict[str, Any] = {}
    scores = await meme_namespace["_rerank_meme_candidates_for_context"](
        visual, search_intent="根据最新对话选择一张自然反应；没有合适的可以不选", history=rows[-8:], candidates=candidates, candidate_grid=grid,
        review_metadata=review_metadata,
    )
    assert review_metadata.get("reviewed"), "Visual review did not complete successfully"
    assert all(mid in ids for mid, _ in scores)
    record = {
        "scope": "real-model helper replay; no Agent rollout or message delivery",
        "at": datetime.now().isoformat(), "group": group, "human_sources": len(sources),
        "flash_model": cfg.flash_model, "visual_model": cfg.vision_model or cfg.chat_model,
        "expressions": [unit.model_dump(mode="json") for unit in bank.expressions],
        "selections": selections, "visible_candidates": len(ids), "visual_scores": scores, "visual_review": review_metadata,
    }
    (output / "results.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: record[key] for key in ("scope", "human_sources", "flash_model", "visual_model", "visible_candidates", "visual_scores")} | {
        "learned_units": len(bank.expressions), "selected_turns": sum(bool(item["selected_context"]) for item in selections),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", default="758450633")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    print(json.dumps(asyncio.run(replay(args.package_root, args.output, args.group)), ensure_ascii=False))
