import sys
import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest


def test_seatbelt_profile_escapes_workspace(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.sandbox_platform import seatbelt_profile

    profile = seatbelt_profile(tmp_path)
    assert "(deny file-write*)" in profile
    assert '(literal "/dev/null")' in profile
    assert str(tmp_path.resolve()).replace("\\", "\\\\") in profile


def test_windows_explicit_missing_runner_fails_closed(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.sandbox_platform import SandboxUnavailable, windows_runner

    with pytest.raises(SandboxUnavailable, match="未降级"):
        windows_runner("node", str(tmp_path / "missing.js"))


@pytest.mark.parametrize("target_platform", ["linux", "darwin", "win32"])
def test_sandbox_dispatch_keeps_command_as_arguments(tmp_path, monkeypatch, target_platform):
    from nonebot_plugin_ai_groupmate.agent import sandbox_platform as module

    monkeypatch.setattr(module.sys, "platform", target_platform)
    monkeypatch.setattr(module, "supported_platform", lambda: True)
    monkeypatch.setattr(module.shutil, "which", lambda name: "node.exe")
    monkeypatch.setattr(module, "windows_runner", lambda *args: tmp_path / "runner.js")
    monkeypatch.setattr(module.Path, "is_file", lambda self: True)
    argv = ["python", "a path with spaces.py", "literal;$value"]
    wrapped = module.sandbox_command(argv, tmp_path, windows_temp=tmp_path / "private")
    assert wrapped[-3:] == argv
    assert len(wrapped) > len(argv)
    if target_platform == "darwin":
        assert wrapped[0] == str(Path("/usr/bin/sandbox-exec"))
    elif target_platform == "win32":
        assert "workspace-write" in wrapped
    else:
        assert "-I" in wrapped


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows ACL sandbox")
@pytest.mark.parametrize("language", ["javascript", "shell"])
async def test_windows_sandbox_native_runtimes(language):
    from nonebot_plugin_ai_groupmate.agent.sandbox_platform import SandboxUnavailable, windows_runner
    from nonebot_plugin_ai_groupmate.agent.execution_backend import run_local, decode_files

    node = shutil.which("node")
    if not node:
        pytest.skip("Node not installed")
    try:
        runner = windows_runner(node)
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    code = "require('fs').writeFileSync('out/native.txt', 'ok')" if language == "javascript" else "[System.IO.File]::WriteAllText((Join-Path (Get-Location) 'out/native.txt'), 'ok')"
    result = await run_local({"language": language, "code": code}, sandbox=True, timeout=30, sandbox_windows_runner=str(runner))
    assert result["exit_code"] == 0, result["stderr"]
    assert decode_files(result["files"], limit=100) == [("native.txt", b"ok")]


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows ACL sandbox")
async def test_windows_real_sandbox_restricts_mutations_and_descendants(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.sandbox_platform import SandboxUnavailable, windows_runner
    from nonebot_plugin_ai_groupmate.agent.execution_backend import run_local, decode_files, _prepare_windows_sandbox_dirs

    node = shutil.which("node")
    if not node:
        pytest.skip("Node not installed")
    try:
        runner = windows_runner(node)
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    target = tmp_path / "protected.txt"
    target.write_text("unchanged")
    # Make this private pytest fixture readable to the lowered token as well.
    await _prepare_windows_sandbox_dirs([tmp_path])
    code = f"""
from pathlib import Path
import os, subprocess, sys
target = Path({str(target)!r})
assert target.read_text() == 'unchanged'
for operation in [lambda: target.write_text('bad'), lambda: os.truncate(target, 0), lambda: target.unlink(), lambda: target.rename(target.with_name('renamed'))]:
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise AssertionError('outside mutation allowed')
child = subprocess.run([sys.executable, '-c', "from pathlib import Path; Path(" + repr(str(target)) + ").write_text('bad')"], capture_output=True)
assert child.returncode != 0
Path('out/result.txt').write_text('confined')
"""
    result = await run_local({"language": "python", "code": code}, timeout=30, sandbox=True, sandbox_windows_runner=str(runner))
    assert result["exit_code"] == 0, result["stderr"]
    assert decode_files(result["files"], limit=100) == [("result.txt", b"confined")]
    assert target.read_text() == "unchanged"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS Seatbelt")
async def test_macos_real_sandbox_restricts_outside_writes(tmp_path):
    from nonebot_plugin_ai_groupmate.agent.execution_backend import run_local

    target = tmp_path / "protected.txt"
    target.write_text("unchanged")
    code = f"""
from pathlib import Path
target = Path({str(target)!r})
try:
    target.write_text('bad')
except PermissionError:
    Path('out/ok.txt').write_text('confined')
else:
    raise AssertionError('outside write allowed')
"""
    result = await run_local({"language": "python", "code": code}, sandbox=True)
    assert result["exit_code"] == 0, result["stderr"]
    assert target.read_text() == "unchanged"


async def test_sandbox_unsupported_platform_never_executes(monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import sandbox_platform, execution_backend

    monkeypatch.setattr(sandbox_platform, "supported_platform", lambda: False)

    async def forbidden(*args, **kwargs):
        pytest.fail("unsupported sandbox executed a command")

    monkeypatch.setattr(execution_backend, "_command", forbidden)
    with pytest.raises(execution_backend.ExecutionError, match="未降级"):
        await execution_backend.run_local({"language": "python", "code": "print('must not execute')"}, sandbox=True)


@pytest.mark.parametrize("failure", ["abi", "create", "add", "privileges", "restrict"])
def test_sandbox_setup_failure_does_not_exec(tmp_path, monkeypatch, failure):
    from nonebot_plugin_ai_groupmate.agent import sandbox_launcher as module

    class Libc:
        def syscall(self, number, *args):
            if number == module.CREATE_RULESET and args[-1] == 1:
                return 2 if failure == "abi" else 3
            if number == module.CREATE_RULESET:
                return -1 if failure == "create" else 40
            if number == module.ADD_RULE:
                return -1 if failure == "add" else 0
            return -1 if failure == "restrict" else 0

        def prctl(self, *args):
            return -1 if failure == "privileges" else 0

    instance = Libc()
    # ctypes function objects have writable restype attributes.
    class Syscall:
        restype = ctypes.c_long

        def __call__(self, *args):
            return instance.syscall(*args)

    class Library:
        syscall = Syscall()
        prctl = instance.prctl

    monkeypatch.setattr(module, "supported_platform", lambda: True)
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *args, **kwargs: Library())
    monkeypatch.setattr(module.os, "O_PATH", 0, raising=False)
    monkeypatch.setattr(module.os, "O_CLOEXEC", 0, raising=False)
    monkeypatch.setattr(module.os, "open", lambda *args: 41)
    monkeypatch.setattr(module.os, "close", lambda *args: None)
    monkeypatch.setattr(module.os, "execv", lambda *args: pytest.fail("setup failure executed user code"))
    monkeypatch.setattr(module.sys, "argv", ["launcher", str(tmp_path), "--", sys.executable, "-c", "pass"])
    assert module.main() == 125


@pytest.mark.skipif(sys.platform != "linux", reason="real Landlock enforcement requires Linux")
def test_linux_sandbox_denies_outside_writes_and_child_writes(tmp_path):
    from nonebot_plugin_ai_groupmate.agent import sandbox_launcher as module

    root = tmp_path / "workspace"
    root.mkdir()
    protected = tmp_path / "protected.txt"
    protected.write_text("unchanged")
    code = f"""
from pathlib import Path
import os, sys, subprocess
root = Path({str(root)!r})
protected = Path({str(protected)!r})
(root / 'ok.txt').write_text('inside allowed')
assert protected.read_text() == 'unchanged'  # This is write restriction, not read confinement.
(root / 'alias').symlink_to(protected)
operations = [lambda: protected.write_text('bad'), lambda: os.truncate(protected, 0),
              lambda: protected.unlink(), lambda: protected.rename(root / 'stolen'),
              lambda: (root / 'alias').write_text('bad'),
              lambda: (protected.parent / 'new-directory').mkdir()]
for operation in operations:
    try:
        operation()
    except OSError as error:
        assert error.errno in (1, 13, 18), error
    else:
        raise AssertionError('outside mutation was allowed')
child = subprocess.run([sys.executable, '-c', "from pathlib import Path; Path(" + repr(str(protected)) + ").write_text('bad')"], capture_output=True)
assert child.returncode != 0
assert protected.read_text() == 'unchanged'
print('enforcement verified')
"""
    result = subprocess.run([sys.executable, "-I", "-S", str(Path(module.__file__).resolve()), str(root), "--", sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    if result.returncode == 125:
        assert not (root / "ok.txt").exists()
        assert "Sandbox launch refused" in result.stderr
        pytest.skip("Landlock unavailable or blocked by runner policy: " + result.stderr[:250])
    assert result.returncode == 0, result.stderr
    assert "enforcement verified" in result.stdout
    assert protected.read_text() == "unchanged"


async def test_sandbox_tool_selects_confined_backend(tmp_path, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import execution_tools as module
    from nonebot_plugin_ai_groupmate.config import ScopedConfig
    from nonebot_plugin_ai_groupmate.agent.file_workspace import FileWorkspace

    calls = []

    async def local(request, **kwargs):
        calls.append(kwargs)
        return {"exit_code": 0, "files": []}

    monkeypatch.setattr(module, "run_local", local)
    store = FileWorkspace(tmp_path, session_id="s", user_id="u", bot_id="b", is_private=False)
    executor = module.create_execution_tool(ScopedConfig(agent_execution_backend="sandbox"), store, "s", None)
    await executor.ainvoke({"code": "print(1)"})
    assert calls == [{"timeout": 20, "sandbox": True, "sandbox_windows_runner": ""}]
