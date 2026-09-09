from __future__ import annotations

import uuid
import datetime
from typing import cast
from unittest.mock import MagicMock

import pytest
from nonebot.adapters import Bot, Event


def test_web_detection_defaults_stay_inside_free_tier():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    config = ScopedConfig()

    assert config.google_web_detection_enabled is False
    assert config.google_web_detection_monthly_limit == 900
    assert config.google_web_detection_cache_days == 30
    assert config.google_web_detection_max_results == 5


@pytest.mark.parametrize(
    "text",
    [
        "帮我以图搜图",
        "反向搜图看看",
        "帮我找一下这张图的出处",
        "这图片哪来的",
        "查下这张图",
        "帮我找原图",
        "这个图片图源是什么",
        "miyuki找一下这个图的出处",
        "miyuki你用搜图试试这张图",
    ],
)
def test_explicit_web_image_search_detection_accepts_reverse_searches(text: str):
    from nonebot_plugin_ai_groupmate.agent.web_image_search import (
        is_explicit_web_image_search_request,
    )

    assert is_explicit_web_image_search_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "帮我搜图 初音未来",
        "描述一下这张图",
        "这张图好看吗",
        "给我找一张猫图",
        "帮我整理一下 Arcaea 的剧情",
    ],
)
def test_explicit_web_image_search_detection_rejects_other_image_requests(text: str):
    from nonebot_plugin_ai_groupmate.agent.web_image_search import (
        is_explicit_web_image_search_request,
    )

    assert is_explicit_web_image_search_request(text) is False


def test_normalize_web_detection_response_bounds_and_sanitizes_results():
    from nonebot_plugin_ai_groupmate.agent.web_image_search import (
        normalize_web_detection_response,
    )

    result = normalize_web_detection_response(
        {
            "bestGuessLabels": [
                {"label": " illustration ", "languageCode": "en"},
                {"label": "second"},
            ],
            "webEntities": [
                {"entityId": "/m/example", "score": 0.75, "description": "角色"}
            ],
            "pagesWithMatchingImages": [
                {
                    "url": "https://example.com/source",
                    "pageTitle": "<b>Original</b> &amp; Source",
                    "score": 0.9,
                },
                {"url": "javascript:alert(1)", "pageTitle": "bad"},
            ],
            "fullMatchingImages": [
                {"url": "https://cdn.example.com/a.png", "score": 1.0},
                {"url": "https://cdn.example.com/b.png"},
            ],
            "partialMatchingImages": [{"url": "file:///secret.png"}],
            "visuallySimilarImages": [{"url": "https://img.example.com/similar.jpg"}],
        },
        max_results=1,
    )

    assert result["best_guess_labels"] == [
        {"label": "illustration", "language_code": "en"}
    ]
    assert result["web_entities"] == [
        {"description": "角色", "entity_id": "/m/example", "score": 0.75}
    ]
    assert result["pages_with_matching_images"] == [
        {
            "url": "https://example.com/source",
            "title": "Original & Source",
            "score": 0.9,
        }
    ]
    assert result["full_matching_images"] == [
        {"url": "https://cdn.example.com/a.png", "score": 1.0}
    ]
    assert result["partial_matching_images"] == []


@pytest.mark.asyncio
async def test_web_detection_cache_does_not_consume_a_second_unit():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import GoogleWebDetectionUsage
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.web_image_search import (
        perform_web_detection,
    )

    image_hash = uuid.uuid4().hex.ljust(64, "0")
    calls = 0

    async def provider(_: bytes) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "best_guess_labels": [{"label": "test image"}],
            "pages_with_matching_images": [],
        }

    config = ScopedConfig(
        google_web_detection_enabled=True,
        google_web_detection_monthly_limit=10,
    )
    now = datetime.datetime(2098, 1, 3, 12, 0, 0)
    async with get_session() as db_session:
        db_session.sync_session.expire_on_commit = True
        first = await perform_web_detection(
            db_session,
            image_hash,
            b"same-image",
            config,
            provider=provider,
            now=now,
        )
        second = await perform_web_detection(
            db_session,
            image_hash,
            b"same-image",
            config,
            provider=provider,
            now=now + datetime.timedelta(days=1),
        )
        usage = await db_session.get(GoogleWebDetectionUsage, "2098-01")

    assert calls == 1
    assert first.cached is False
    assert second.cached is True
    assert first.used_count == second.used_count == 1
    assert usage is not None
    assert usage.used_count == 1


@pytest.mark.asyncio
async def test_web_detection_monthly_limit_blocks_provider_call():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.web_image_search import (
        WebDetectionQuotaExceeded,
        perform_web_detection,
    )

    calls = 0

    async def provider(_: bytes) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"best_guess_labels": []}

    config = ScopedConfig(
        google_web_detection_enabled=True,
        google_web_detection_monthly_limit=1,
    )
    now = datetime.datetime(2098, 2, 3, 12, 0, 0)
    async with get_session() as db_session:
        await perform_web_detection(
            db_session,
            uuid.uuid4().hex.ljust(64, "1"),
            b"first-image",
            config,
            provider=provider,
            now=now,
        )
        with pytest.raises(WebDetectionQuotaExceeded) as exc_info:
            await perform_web_detection(
                db_session,
                uuid.uuid4().hex.ljust(64, "2"),
                b"second-image",
                config,
                provider=provider,
                now=now,
            )

    assert calls == 1
    assert exc_info.value.used_count == 1
    assert exc_info.value.monthly_limit == 1


@pytest.mark.asyncio
async def test_source_image_prefers_reply_then_current_user():
    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.model import ChatHistory, MediaStorage
    from nonebot_plugin_ai_groupmate.agent.web_image_search import (
        _find_source_media,
    )

    unique = uuid.uuid4().hex
    session_id = f"group-{unique}"
    async with get_session() as db_session:
        alice_media = MediaStorage(
            file_hash=uuid.uuid4().hex.ljust(64, "a"),
            file_path="alice.jpg",
            description="Alice image",
        )
        bob_media = MediaStorage(
            file_hash=uuid.uuid4().hex.ljust(64, "b"),
            file_path="bob.jpg",
            description="Bob image",
        )
        db_session.add_all([alice_media, bob_media])
        await db_session.flush()
        db_session.add_all(
            [
                ChatHistory(
                    session_id=session_id,
                    user_id="alice",
                    content_type="image",
                    content="id: alice-image\nalice.jpg",
                    created_at=datetime.datetime(2098, 3, 1, 10, 0, 0),
                    user_name="Alice",
                    media_id=alice_media.media_id,
                ),
                ChatHistory(
                    session_id=session_id,
                    user_id="bob",
                    content_type="image",
                    content="id: bob-image\nbob.jpg",
                    created_at=datetime.datetime(2098, 3, 1, 11, 0, 0),
                    user_name="Bob",
                    media_id=bob_media.media_id,
                ),
            ]
        )
        await db_session.commit()

        replied = await _find_source_media(
            db_session,
            session_id,
            "bob",
            None,
            "alice-image",
        )
        own_latest = await _find_source_media(
            db_session,
            session_id,
            "alice",
            None,
            None,
        )
        invalid_explicit = await _find_source_media(
            db_session,
            session_id,
            "alice",
            "missing-id",
            None,
        )
        invalid_reply = await _find_source_media(db_session, session_id, "bob", None, "missing-reply")

    assert replied is not None
    assert replied[1].media_id == alice_media.media_id
    assert own_latest is not None
    assert own_latest[1].media_id == alice_media.media_id
    assert invalid_explicit is None
    assert invalid_reply is None


@pytest.mark.asyncio
async def test_duplicate_image_preserves_both_bot_message_ids(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from sqlalchemy import Select
    from nonebot_plugin_orm import get_session
    from nonebot_plugin_uninfo import Uninfo
    from nonebot_plugin_alconna.uniseg import Image

    import nonebot_plugin_ai_groupmate as plugin
    from nonebot_plugin_ai_groupmate.model import ChatHistory
    from nonebot_plugin_ai_groupmate.agent.web_image_search import _find_source_media

    image_bytes = uuid.uuid4().bytes
    async def fetch(*args):
        return image_bytes

    monkeypatch.setattr(plugin, "image_fetch", fetch)
    monkeypatch.setattr(plugin, "check_and_compress_image_bytes", lambda data, **kw: data)
    monkeypatch.setattr(plugin, "pic_dir", tmp_path)
    group = uuid.uuid4().hex
    session = SimpleNamespace(scene=SimpleNamespace(id=group), user=SimpleNamespace(id="alice"))
    async with get_session() as db:
        for message_id in ("bot-one-id", "bot-two-id", "bot-two-id"):
            await plugin.process_image_message(db, Image(id="image.jpg"), MagicMock(spec=Event), MagicMock(spec=Bot), {}, cast(Uninfo, session), "Alice", f"id: {message_id}\n")
        rows = (await db.execute(Select(ChatHistory).where(ChatHistory.session_id == group))).scalars().all()
        assert len(rows) == 1
        assert rows[0].content.count("id: bot-two-id\n") == 1
        first = await _find_source_media(db, group, "alice", None, "bot-one-id")
        second = await _find_source_media(db, group, "alice", None, "bot-two-id")
        assert first is not None
        assert second is not None
        assert first[1].media_id == second[1].media_id
        assert rows[0].content.splitlines()[-1] == first[1].file_path


def test_source_image_reader_rejects_path_escape(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.web_image_search import (
        WebDetectionError,
        _read_source_image,
    )

    pic_dir = tmp_path / "pics"
    pic_dir.mkdir()
    (pic_dir / "inside.jpg").write_bytes(b"image")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")

    assert _read_source_image(pic_dir, "inside.jpg") == b"image"
    with pytest.raises(WebDetectionError) as exc_info:
        _read_source_image(pic_dir, "../outside.jpg")

    assert exc_info.value.reason_code == "source_file_not_found"


def test_settings_page_exposes_google_web_detection_fields():
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.settings_ui import render_settings_page

    html = render_settings_page(
        ScopedConfig(),
        ScopedConfig(),
        overridden_fields=set(),
        pending_restart_fields=set(),
        dashboard_path="/usage",
        settings_path="/usage/settings",
    )

    assert "Google 反向搜图" in html
    assert 'data-setting="google_web_detection_enabled"' in html
    assert 'data-setting="google_web_detection_monthly_limit"' in html


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_meme", [False, True])
@pytest.mark.parametrize(
    ("request_text", "expected_visible"),
    [
        ("帮我找一下这张图的出处", True),
        ("miyuki找一下这个的出处", True),
        ("miyuki你用搜图试试这张图", True),
        ("给我找一张猫图", True),
        ("你搜图搜到了什么", True),
    ],
)
async def test_agent_exposes_search_and_evidence_tools_without_keyword_gate(
    monkeypatch,
    request_text: str,
    expected_visible: bool,
    explicit_meme: bool,
):
    from nonebot_plugin_ai_groupmate import agent

    captured_base_tools: list[str] = []
    captured_factory_options: dict[str, object] = {}

    class FakeSession:
        async def commit(self):
            return None

    class FakeEvent:
        def get_plaintext(self) -> str:
            return request_text

    class DummyTool:
        def __init__(self, name: str):
            self.name = name

    async def empty_context(*args, **kwargs) -> str:
        return ""

    async def no_extensions(*args, **kwargs):
        return [], [], []

    def fake_reverse_tool(*args, **kwargs):
        captured_factory_options.update(kwargs)
        return DummyTool("reverse_image_search")

    def fake_build_chat_graph(model, tools, system_prompt, **kwargs):
        captured_base_tools.extend(tool.name for tool in kwargs["base_tools"])
        return object()

    monkeypatch.setattr(agent, "get_user_relation_context", empty_context)
    monkeypatch.setattr(agent, "get_group_context", empty_context)
    monkeypatch.setattr(agent, "get_recent_relations_context", empty_context)
    monkeypatch.setattr(agent, "build_registered_agent_extensions", no_extensions)
    monkeypatch.setattr(agent, "get_chat_model", lambda: object())
    monkeypatch.setattr(agent, "get_flash_model", lambda: object())
    monkeypatch.setattr(agent, "build_chat_graph", fake_build_chat_graph)
    monkeypatch.setattr(agent, "create_reverse_image_search_tool", fake_reverse_tool)
    monkeypatch.setattr(agent.plugin_config, "google_web_detection_enabled", True)

    await agent.create_chat_graph(
        FakeSession(),
        "group-1",
        None,
        "user-1",
        "Alice",
        history=[],
        event=cast(Event, FakeEvent()),
        is_private=False,
        reply_to_id="quoted-image-id",
        meme_required=explicit_meme,
        proactive_meme_only=explicit_meme,
    )

    assert ("reverse_image_search" in captured_base_tools) is expected_visible
    assert "get_last_image_search_result" in captured_base_tools
    assert "reply_user" in captured_base_tools
    if expected_visible:
        assert captured_factory_options["reply_to_id"] == "quoted-image-id"


@pytest.mark.parametrize("text", ["miyuki找一下这个图的出处", "miyuki你用搜图试试这张图", "找这个表情包的出处"])
def test_reverse_search_does_not_enter_meme_only_mode(text):
    import nonebot_plugin_ai_groupmate as plugin

    assert not plugin._is_explicit_meme_request(text)


def test_implicit_source_search_requires_attachment_context():
    from nonebot_plugin_ai_groupmate.agent.web_image_search import is_explicit_web_image_search_request

    assert not is_explicit_web_image_search_request("找一下这个的出处")
    assert is_explicit_web_image_search_request("找一下这个的出处", has_image_context=True)


@pytest.mark.asyncio
async def test_service_account_auth_does_not_require_oauth_network(monkeypatch):
    from unittest.mock import Mock

    from google.oauth2.service_account import Credentials

    from nonebot_plugin_ai_groupmate.agent import web_image_search
    from nonebot_plugin_ai_groupmate.config import ScopedConfig

    signer = Mock(key_id="test-key")
    signer.sign.return_value = b"test-signature"
    credentials = Credentials(
        signer, "bot@test-project.iam.gserviceaccount.com",
        "https://oauth2.googleapis.com/token",
        scopes=[web_image_search.GOOGLE_CLOUD_PLATFORM_SCOPE],
    )
    monkeypatch.setattr(web_image_search.google.auth, "load_credentials_from_file", lambda *a, **kw: (credentials, "test-project"))
    request = Mock(side_effect=AssertionError("Service account authentication must not call OAuth"))
    monkeypatch.setattr(web_image_search, "GoogleAuthRequest", lambda: request)
    web_image_search._load_google_credentials.cache_clear()
    try:
        auth = await web_image_search._prepare_google_auth(ScopedConfig(google_web_detection_credentials_path="test-service-account.json"))
        assert auth.project_id == "test-project"
        assert auth.access_token
        request.assert_not_called()
    finally:
        web_image_search._load_google_credentials.cache_clear()


@pytest.mark.asyncio
async def test_last_search_persists_evidence_and_failure_without_cross_user_leaks():
    import json

    from nonebot_plugin_orm import get_session

    from nonebot_plugin_ai_groupmate.agent.tool_results import tool_failure, tool_success
    from nonebot_plugin_ai_groupmate.agent.web_image_search import save_last_image_search, create_last_image_search_tool

    group = uuid.uuid4().hex
    success = tool_success("web_detection_matches_found", "matches", data={"source_media_id": 123, "result": {"full_matching_images": [{"url": "https://example.com/source.jpg"}]}})
    async with get_session() as db:
        await save_last_image_search(db, group, "alice", success, datetime.datetime(2098, 1, 2))
    async with get_session() as db:
        reader = create_last_image_search_tool(db, group, "alice")
        result = json.loads(await reader.ainvoke({}))
        assert result["data"]["search_result"] == json.loads(success)
        other_user = json.loads(await create_last_image_search_tool(db, group, "bob").ainvoke({}))
        other_group = json.loads(await create_last_image_search_tool(db, group + "other", "alice").ainvoke({}))
        assert other_user["ok"] is False
        assert other_group["ok"] is False
        failure = tool_failure("provider_unavailable", "failed", retryable=True)
        await save_last_image_search(db, group, "alice", failure, datetime.datetime(2098, 1, 3))
        await save_last_image_search(db, group, "alice", success, datetime.datetime(2098, 1, 1))
        result = json.loads(await reader.ainvoke({}))
        assert result["data"]["search_result"]["status"] == "failed"
