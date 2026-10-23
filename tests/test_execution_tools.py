import io
import os
import json
import base64
import shutil
import asyncio
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image


async def test_local_python_round_trip_and_environment(tmp_path, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent.execution_backend import run_local, decode_files

    monkeypatch.setenv("DS_API_KEY", "must-not-inherit")
    result = await run_local({"language": "python", "code": "from pathlib import Path\nimport os\nassert 'DS_API_KEY' not in os.environ\nPath('out/result.txt').write_bytes(Path('inputs/input.txt').read_bytes()+b' done')\nprint('ok')",
                              "files": [{"name": "input.txt", "data": base64.b64encode(b"test").decode()}]})
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "ok"
    assert decode_files(result["files"], limit=100) == [("result.txt", b"test done")]


@pytest.mark.parametrize("language", ["javascript", "shell"])
async def test_local_native_runtimes(language):
    from nonebot_plugin_ai_groupmate.agent.execution_backend import run_local, decode_files

    if language == "javascript" and not shutil.which("node"):
        pytest.skip("Node.js not installed")
    code = "require('fs').writeFileSync('out/result.txt', 'native')" if language == "javascript" else (
        "[System.IO.File]::WriteAllText((Join-Path (Get-Location) 'out/result.txt'), 'native')" if os.name == "nt" else "printf native > out/result.txt")
    result = await run_local({"language": language, "code": code})
    assert result["exit_code"] == 0, result["stderr"]
    assert decode_files(result["files"], limit=100) == [("result.txt", b"native")]


@pytest.mark.parametrize("cancel", [False, True])
async def test_local_timeout_and_cancellation(tmp_path, cancel):
    from nonebot_plugin_ai_groupmate.agent.execution_backend import ExecutionError, run_local

    marker = tmp_path / "started"
    code = f"from pathlib import Path\nimport time\nPath({str(marker)!r}).write_text(str(Path.cwd()))\ntime.sleep(30)"
    task = asyncio.create_task(run_local({"language": "python", "code": code}, timeout=2))
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists()
    directory = marker.read_text()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else ExecutionError):
        await task
    assert not os.path.exists(directory)


async def test_local_output_links_rejected():
    from nonebot_plugin_ai_groupmate.agent.execution_backend import ExecutionError, run_local

    if os.name == "nt":
        pytest.skip("Creating symlinks may require Windows developer mode")
    with pytest.raises(ExecutionError, match="符号链接"):
        await run_local({"language": "python", "code": "from pathlib import Path\nPath('out/link').symlink_to('/etc/passwd')"})


async def test_local_tool_publishes_real_image_to_workspace(tmp_path):
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.file_content import image_preview
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace
    from nonebot_plugin_ai_groupmate.agent.execution_tools import create_execution_tool

    store = FileWorkspace(tmp_path, session_id="s", user_id="u", bot_id="b", is_private=False)
    executor = create_execution_tool(ScopedConfig(agent_execution_backend="local"), store, "s", None)
    result = json.loads(await executor.ainvoke({"code": "from PIL import Image\nImage.new('RGB', (24, 16), 'blue').save('out/actual.png')"}))
    assert result["status"] == "succeeded"
    artifact_id = result["data"]["files"][0]["file_id"]
    details, uri = image_preview(store.read(artifact_id))
    assert details["width"] == 24
    assert uri.startswith("data:image/jpeg;base64,")


def output_tar(name="result.txt", content=b"result", *, symlink=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        item = tarfile.TarInfo(name)
        if symlink:
            item.type = tarfile.SYMTYPE
            item.linkname = "/etc/passwd"
            archive.addfile(item)
        else:
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
    return output.getvalue()


@pytest.mark.parametrize(("name", "symlink"), [("../secret", False), ("/etc/passwd", False), ("link", True)])
def test_execution_outputs_reject_paths_and_links(name, symlink):
    from nonebot_plugin_ai_groupmate.agent.execution_backend import ExecutionError, output_files

    with pytest.raises(ExecutionError):
        output_files(output_tar(name, symlink=symlink))


async def test_docker_job_is_offline_bounded_and_always_removed(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import execution_backend as module

    calls = []

    async def command(*args, **kwargs):
        calls.append((args, kwargs))
        if args[:2] == ("docker", "cp"):
            return 1, b"", b"Error response from daemon: container rootfs is marked read-only"
        if args[:2] == ("docker", "exec") and "tar" in args:
            return (0, b"", b"") if "stdin" in kwargs else (0, output_tar("./result.txt"), b"")
        if args[:2] == ("docker", "exec"):
            return 3, b"done", b"program warning"
        return 0, b"", b""

    monkeypatch.setattr(module, "_command", command)
    result = await module.run_docker({"code": "print('done')", "language": "python"}, image="test-image")
    assert result["stdout"] == "done"
    assert result["exit_code"] == 3
    assert result["stderr"] == "program warning"
    assert module.decode_files(result["files"], limit=100) == [("result.txt", b"result")]
    create = calls[0][0]
    for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=512m", "--pids-limit=128", "--pull=never", "--tmpfs=/workspace:rw,nosuid,nodev,size=64m"):
        assert flag in create
    assert not any("mount" in flag or "socket" in flag for flag in create)
    assert calls[-1][0][:4] == ("docker", "rm", "--force", "--volumes")
    transfer, options = next((args, kwargs) for args, kwargs in calls if "stdin" in kwargs)
    assert transfer[:3] == ("docker", "exec", "-i")
    assert transfer[4:] == ("tar", "-xof", "-", "-C", "/workspace")
    inputs = options["stdin"]
    with tarfile.open(fileobj=io.BytesIO(inputs)) as archive:
        assert set(archive.getnames()) == {"inputs", "out", "program"}
    collect, options = next((args, kwargs) for args, kwargs in calls if kwargs.get("strict"))
    assert collect[:2] == ("docker", "exec")
    assert collect[3:] == ("tar", "-cf", "-", "-C", "/workspace/out", ".")
    assert options["limit"] == module.MAX_OUTPUT_BYTES + 1024 * 1024


@pytest.mark.parametrize("stage", ["input", "output"])
async def test_docker_transfer_errors_include_bounded_diagnostics_and_cleanup(monkeypatch, stage):
    from nonebot_plugin_ai_groupmate.agent import execution_backend as module

    calls = []
    diagnostic = b"tar: transfer failed: \xff" + b"x" * 500

    async def command(*args, **kwargs):
        calls.append(args)
        if (stage == "input" and "stdin" in kwargs) or (stage == "output" and kwargs.get("strict")):
            return 2, b"", diagnostic
        return 0, b"", b""

    monkeypatch.setattr(module, "_command", command)
    message = "无法传入执行文件。" if stage == "input" else "无法收集输出文件，请勿删除 out 目录。"
    with pytest.raises(module.ExecutionError) as error:
        await module.run_docker({"code": "pass", "language": "python"}, image="test-image")
    assert str(error.value) == message + diagnostic.decode("utf-8", "replace")[:200]
    assert calls[-1][:4] == ("docker", "rm", "--force", "--volumes")


@pytest.mark.parametrize("cancel", [False, True])
async def test_docker_timeout_or_cancellation_removes_container(monkeypatch, cancel):
    from nonebot_plugin_ai_groupmate.agent import execution_backend as module

    started = asyncio.Event()
    removed = []

    async def command(*args, **kwargs):
        if args[:2] == ("docker", "exec"):
            started.set()
            await asyncio.sleep(10)
        if args[:2] == ("docker", "rm"):
            removed.append(args[-1])
        return 0, b"", b""

    monkeypatch.setattr(module, "_command", command)
    task = asyncio.create_task(module.run_docker({"code": "pass", "language": "python"}, image="test", timeout=0.05))
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else module.ExecutionError):
        await task
    assert len(removed) == 1


async def test_remote_uses_same_protocol_and_does_not_follow_redirect(monkeypatch):
    import httpx

    from nonebot_plugin_ai_groupmate.agent import execution_backend as module

    def respond(request):
        assert request.headers["authorization"] == "Bearer test-token"
        assert json.loads(request.content)["code"] == "print(1)"
        return httpx.Response(200, json={"exit_code": 0, "stdout": "1", "stderr": "", "files": []})

    original = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    result = await module.run_remote({"language": "python", "code": "print(1)"}, url="http://executor:8091", token="test-token", timeout=2)
    assert result["stdout"] == "1"


async def test_remote_gateway_authentication_and_round_trip(monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer

    import httpx

    from nonebot_plugin_ai_groupmate.agent import execution_service as service
    from nonebot_plugin_ai_groupmate.agent.execution_backend import run_remote

    execute = AsyncMock(return_value={"exit_code": 0, "stdout": "ok", "stderr": "", "files": []})
    monkeypatch.setattr(service, "run_docker", execute)
    server = ThreadingHTTPServer(("127.0.0.1", 0), service.make_handler("test-service-token", "test-image", 2))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(url + "/execute", json={"code": "pass"})
            assert response.status_code == 403
            execute.assert_not_called()
        result = await run_remote({"language": "python", "code": "print('ok')"}, url=url, token="test-service-token", timeout=2)
        assert result["stdout"] == "ok"
        execute.assert_awaited_once_with({"language": "python", "code": "print('ok')"}, image="test-image", timeout=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def test_code_output_can_be_seen_revised_and_sent(tmp_path, monkeypatch):
    from nonebot_plugin_alconna import Target, UniMessage
    from langchain_core.messages import AIMessage, HumanMessage

    from nonebot_plugin_ai_groupmate.agent import execution_tools as module
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.graph import build_chat_graph
    from nonebot_plugin_ai_groupmate.agent.file_tools import create_file_tools
    from nonebot_plugin_ai_groupmate.agent.common_tools import finish
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    store = FileWorkspace(tmp_path, session_id="s", user_id="u", bot_id="b", is_private=False)
    seed = store.save(b"seed", "input.txt")
    sent = []
    executions = []

    async def execute(request, **kwargs):
        nonlocal first_image
        executions.append(request)
        assert base64.b64decode(request["files"][0]["data"]) == (b"seed" if len(executions) == 1 else first_image)
        image = io.BytesIO()
        Image.new("RGB", (20, 20), "red" if len(executions) == 1 else "blue").save(image, format="PNG")
        if len(executions) == 1:
            first_image = image.getvalue()
        return {"exit_code": 0, "stdout": "", "stderr": "", "files": [{"name": "drawing.png", "data": base64.b64encode(image.getvalue()).decode()}]}

    first_image = b""
    monkeypatch.setattr(module, "run_docker", execute)

    async def send(message, **kwargs):
        sent.append(message[0].path.read_bytes())
        return SimpleNamespace(msg_ids=[{"message_id": "1"}])

    monkeypatch.setattr(UniMessage, "send", send)
    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock(), add=lambda row: None)
    files = create_file_tools(db, "s", None, root=tmp_path, user_id="u", bot_id="b", is_private=False, bot_name="bot", send_target=Target("s"))
    executor = module.create_execution_tool(ScopedConfig(agent_execution_backend="docker"), store, "s", None)

    class Model:
        step = 0
        artifact_id = seed.file_id

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, **kwargs):
            results = []
            for message in messages:
                if isinstance(message.content, str):
                    try:
                        parsed = json.loads(message.content)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except ValueError:
                        pass
            if self.step in (1, 3):
                self.artifact_id = results[-1]["data"]["files"][0]["file_id"]
            if self.step == 0:
                name, args = "run_code", {"code": "generate", "file_ids": [self.artifact_id]}
            elif self.step == 1:
                name, args = "read_file", {"file_id": self.artifact_id, "mode": "image"}
            elif self.step == 2:
                assert any(isinstance(m, HumanMessage) and isinstance(m.content, list) and any(isinstance(b, dict) and b.get("type") == "image_url" for b in m.content) for m in messages)
                name, args = "run_code", {"code": "revise", "file_ids": [self.artifact_id]}
            elif self.step == 3:
                name, args = "send_file", {"file_id": self.artifact_id}
            else:
                name, args = "finish", {}
            self.step += 1
            return AIMessage(content="", tool_calls=[{"id": str(self.step), "name": name, "args": args}])

    graph = build_chat_graph(Model(), [executor, *files, finish], "Create, inspect, revise and deliver.")
    await graph.ainvoke({"messages": [HumanMessage(content="draw")], "session_id": "s", "request_id": None})
    assert len(executions) == 2
    assert len(sent) == 1
    assert Image.open(io.BytesIO(sent[0])).getpixel((0, 0)) == (0, 0, 255)
