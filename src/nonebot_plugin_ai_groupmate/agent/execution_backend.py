"""Provider-neutral execution contract. Docker jobs never mount host directories."""

import io
import os
import re
import csv
import sys
import json
import stat
import uuid
import base64
import shutil
import signal
import asyncio
import tarfile
import tempfile
from typing import Literal
from pathlib import Path, PurePosixPath
from contextlib import ExitStack

import httpx

MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_OUTPUT_BYTES = 20 * 1024 * 1024
MAX_FILES = 12
MAX_CODE_BYTES = 128 * 1024
MAX_LOG_BYTES = 16 * 1024
Language = Literal["python", "javascript", "shell"]


class ExecutionError(ValueError):
    pass


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or any(char in name for char in "\\:\x00") or path.is_absolute() or any(part in {"..", ".", ""} for part in name.split("/")) or len(name) > 160:
        raise ExecutionError("文件名必须是工作区内的相对路径，不能包含 .. 或反斜杠。")
    return name


def decode_files(files: list[dict], *, limit: int) -> list[tuple[str, bytes]]:
    if len(files) > MAX_FILES:
        raise ExecutionError("一次最多传递 12 个文件。")
    result = []
    names: set[str] = set()
    total = 0
    for item in files:
        name = safe_name(item["name"])
        if name in names:
            raise ExecutionError("文件名重复，请先重命名。")
        names.add(name)
        data = base64.b64decode(item["data"], validate=True)
        total += len(data)
        if total > limit:
            raise ExecutionError("文件总大小超过执行接口限制。")
        result.append((name, data))
    return result


def validate_request(request: dict) -> None:
    if request.get("language") not in {"python", "javascript", "shell"}:
        raise ExecutionError("只支持 python、javascript、shell。")
    code = request.get("code")
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > MAX_CODE_BYTES:
        raise ExecutionError("代码须非空且不超过 128 KiB。")
    decode_files(request.get("files", []), limit=MAX_INPUT_BYTES)


def input_archive(request: dict) -> bytes:
    validate_request(request)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("inputs", "out"):
            item = tarfile.TarInfo(name)
            item.type = tarfile.DIRTYPE
            item.mode = 0o777
            archive.addfile(item)
        entries = [("program", request["code"].encode()), *[("inputs/" + name, data) for name, data in decode_files(request.get("files", []), limit=MAX_INPUT_BYTES)]]
        for name, data in entries:
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            archive.addfile(item, io.BytesIO(data))
    return buffer.getvalue()


def output_files(data: bytes) -> list[dict]:
    files = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for item in archive:
            name = item.name.removeprefix("./")
            if item.isdir():
                continue
            if not item.isfile():
                raise ExecutionError("输出不能包含符号链接、设备或其他特殊文件。")
            safe_name(name)
            total += item.size
            if item.size > MAX_INPUT_BYTES or total > MAX_OUTPUT_BYTES or len(files) >= MAX_FILES:
                raise ExecutionError("输出最多 12 个文件、单个 10 MiB、总计 20 MiB。")
            handle = archive.extractfile(item)
            if handle is None:
                raise ExecutionError("无法读取执行输出。")
            files.append({"name": name, "data": base64.b64encode(handle.read()).decode()})
    return files


async def _command(*args: str, stdin: bytes | None = None, limit: int = MAX_LOG_BYTES, strict: bool = False,
                   cwd: Path | None = None, env: dict[str, str] | None = None, local: bool = False) -> tuple[int, bytes, bytes]:
    """Drain pipes with bounded memory and reap subprocesses on cancellation."""
    process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                   cwd=cwd, env=env, start_new_session=local and os.name != "nt")

    async def read(stream: asyncio.StreamReader | None) -> bytes:
        data = bytearray()
        if stream is not None:
            while chunk := await stream.read(65536):
                if strict and len(data) + len(chunk) > limit:
                    raise ExecutionError("执行结果超过大小上限。")
                data.extend(chunk[:max(0, limit - len(data))])
        return bytes(data)

    async def write() -> None:
        if process.stdin is not None and stdin is not None:
            try:
                process.stdin.write(stdin)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

    stdout_task = asyncio.create_task(read(process.stdout))
    stderr_task = asyncio.create_task(read(process.stderr))
    writer_task = asyncio.create_task(write())
    tasks = [stdout_task, stderr_task, writer_task]
    try:
        await asyncio.gather(*tasks)
        return await process.wait(), stdout_task.result(), stderr_task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if local and os.name == "nt" and process.returncode is None:
            # Best-effort process-tree cleanup; local execution is not a sandbox.
            try:
                killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F",
                                                              stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await asyncio.wait_for(killer.wait(), timeout=3)
            except (OSError, asyncio.TimeoutError):
                pass
        elif local and os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


async def _prepare_windows_sandbox_dirs(paths: list[Path]) -> None:
    """Give the current user Modify on our fresh private dirs, not ambient users.

    Python 3.13 mkdtemp uses OWNER RIGHTS; DSH's LUA token cannot use that ACE
    to access task files. An explicit user SID ACE survives token lowering;
    the restricting SID and Low integrity still confine writes to granted roots.
    """
    status, stdout, _ = await asyncio.wait_for(_command("whoami.exe", "/user", "/fo", "csv", "/nh"), timeout=5)
    rows = list(csv.reader(io.StringIO(stdout.decode("utf-8", "replace"))))
    sid = rows[0][-1].strip() if rows and rows[0] else ""
    if status or not re.fullmatch(r"S-1-\d+(?:-\d+)+", sid):
        raise ExecutionError("无法确定 Windows 沙箱用户 SID；未执行代码。")
    for path in paths:
        status, _, _ = await asyncio.wait_for(_command("icacls.exe", str(path), "/grant", f"*{sid}:(OI)(CI)(M)", "/Q"), timeout=5)
        if status:
            raise ExecutionError("无法准备 Windows 沙箱临时目录的用户权限；未执行代码。")


async def run_local(request: dict, *, timeout: float = 20, sandbox: bool = False, sandbox_windows_runner: str = "") -> dict:
    """Execute as the Bot OS user. A temporary cwd is NOT a security boundary."""
    validate_request(request)
    if sandbox:
        from .sandbox_platform import supported_platform

        if not supported_platform():
            raise ExecutionError("当前平台不支持 sandbox；未降级为 local。")
    language = request["language"]
    if language == "python":
        command, suffix = [sys.executable, "-u"], ".py"
    elif language == "javascript":
        executable = shutil.which("node")
        if not executable:
            raise ExecutionError("本机未找到 Node.js，请安装后重启 Bot。")
        command, suffix = [executable], ".js"
    elif os.name == "nt":
        executable = shutil.which("pwsh") or shutil.which("powershell")
        if not executable:
            raise ExecutionError("本机未找到 PowerShell。")
        command, suffix = [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-File"], ".ps1"
    else:
        command, suffix = ["/bin/sh"], ".sh"
    # Preserve runtime discovery, not arbitrary Bot/provider configuration.
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
               "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL", "PLAYWRIGHT_BROWSERS_PATH", "FONTCONFIG_PATH"}
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    with ExitStack() as cleanup:
        directory = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="groupmate-local-"))
        root = Path(directory).resolve()
        (root / "inputs").mkdir()
        (root / "out").mkdir()
        for name, data in decode_files(request.get("files", []), limit=MAX_INPUT_BYTES):
            target = root / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        program = root / ("program" + suffix)
        program.write_text(request["code"], encoding="utf-8-sig" if suffix == ".ps1" else "utf-8")
        argv = [*command, str(program)]
        if sandbox:
            private_temp = root / "tmp"
            private_temp.mkdir()
            env.update(TMPDIR=str(private_temp), TMP=str(private_temp), TEMP=str(private_temp), XDG_CACHE_HOME=str(private_temp / "cache"), MPLCONFIGDIR=str(private_temp / "matplotlib"))
            from .sandbox_platform import SandboxUnavailable, sandbox_command

            windows_temp = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix="groupmate-sandbox-temp-"))).resolve() if sys.platform == "win32" else None
            if windows_temp is not None:
                await _prepare_windows_sandbox_dirs([root, windows_temp])
            try:
                argv = sandbox_command(argv, root, windows_temp=windows_temp, runner=sandbox_windows_runner)
            except SandboxUnavailable as error:
                raise ExecutionError(str(error)) from error
        try:
            exit_code, stdout, stderr = await asyncio.wait_for(_command(*argv, cwd=root, env=env, local=True), timeout=timeout)
        except asyncio.TimeoutError:
            raise ExecutionError("本机代码执行超时，已尝试终止执行进程及子进程。") from None
        files = []
        total = 0
        output = root / "out"
        # Reject links (including Windows junctions) before walking or reading.
        def check(path: Path):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ExecutionError("输出不能包含符号链接或目录联接。")
            return info

        check(output)
        for current, directories, names in os.walk(output, followlinks=False):
            for name in directories:
                check(Path(current) / name)
            for name in sorted(names):
                path = Path(current) / name
                info = check(path)
                if not stat.S_ISREG(info.st_mode):
                    raise ExecutionError("输出只能包含普通文件。")
                total += info.st_size
                if info.st_size > MAX_INPUT_BYTES or total > MAX_OUTPUT_BYTES or len(files) >= MAX_FILES:
                    raise ExecutionError("输出最多 12 个文件、单个 10 MiB、总计 20 MiB。")
                with path.open("rb") as handle:
                    content = handle.read(MAX_INPUT_BYTES + 1)
                if len(content) > MAX_INPUT_BYTES:
                    raise ExecutionError("输出文件超过 10 MiB。")
                files.append({"name": safe_name(path.relative_to(output).as_posix()), "data": base64.b64encode(content).decode()})
        return {"exit_code": exit_code, "stdout": stdout.decode("utf-8", "replace"), "stderr": stderr.decode("utf-8", "replace"), "files": files}


async def run_docker(request: dict, *, image: str, timeout: float = 20) -> dict:
    """One fresh container per invocation, no network/secrets/socket/host mounts."""
    archive = input_archive(request)
    container = "groupmate-job-" + uuid.uuid4().hex
    command = {"python": ["python", "/workspace/program"], "javascript": ["node", "/workspace/program"], "shell": ["sh", "/workspace/program"]}[request["language"]]

    async def job() -> dict:
        status, _, _ = await _command("docker", "create", "--pull=never", "--name", container, "--network=none", "--read-only",
                                     "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=512m", "--memory-swap=512m", "--cpus=1", "--pids-limit=128",
                                     "--tmpfs=/tmp:rw,nosuid,nodev,size=128m", "--tmpfs=/workspace:rw,nosuid,nodev,size=64m", "--workdir=/workspace",
                                     "--env=HOME=/tmp", "--env=PYTHONDONTWRITEBYTECODE=1", image, "sleep", str(timeout + 10))
        if status:
            raise ExecutionError("执行容器无法创建；请确认 Docker 可用且已构建执行镜像。")
        status, _, _ = await _command("docker", "start", container)
        if status:
            raise ExecutionError("执行容器无法启动。")
        # docker cp cannot access tmpfs mounts. Extract inside the running container;
        # -o skips ownership restoration because CAP_CHOWN is dropped.
        status, _, transfer_stderr = await _command("docker", "exec", "-i", container, "tar", "-xof", "-", "-C", "/workspace", stdin=archive)
        if status:
            raise ExecutionError("无法传入执行文件。" + transfer_stderr.decode("utf-8", "replace")[:200])
        exit_code, stdout, stderr = await _command("docker", "exec", container, "timeout", "--signal=KILL", str(timeout), *command)
        status, output, transfer_stderr = await _command("docker", "exec", container, "tar", "-cf", "-", "-C", "/workspace/out", ".", limit=MAX_OUTPUT_BYTES + 1024 * 1024, strict=True)
        if status:
            raise ExecutionError("无法收集输出文件，请勿删除 out 目录。" + transfer_stderr.decode("utf-8", "replace")[:200])
        return {"exit_code": exit_code, "stdout": stdout.decode("utf-8", "replace"), "stderr": stderr.decode("utf-8", "replace"), "files": output_files(output)}

    try:
        return await asyncio.wait_for(job(), timeout=timeout)
    except asyncio.TimeoutError:
        raise ExecutionError("代码执行超时，已请求清理执行容器。") from None
    except FileNotFoundError:
        raise ExecutionError("未找到 Docker；请安装或配置远程执行服务。") from None
    finally:
        # The container name is generated locally, never accepted from model input.
        try:
            await asyncio.wait_for(_command("docker", "rm", "--force", "--volumes", container), timeout=5)
        except (OSError, asyncio.TimeoutError):
            pass


async def run_remote(request: dict, *, url: str, token: str, timeout: float) -> dict:
    validate_request(request)
    # This URL belongs to trusted administrator configuration, never model args.
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        async with client.stream("POST", url.rstrip("/") + "/execute", headers={"Authorization": "Bearer " + token}, json=request) as response:
            if response.status_code != 200:
                raise ExecutionError("远程执行服务拒绝请求或不可用，请检查服务配置。")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_OUTPUT_BYTES * 2:
                    raise ExecutionError("远程执行结果过大。")
    result = json.loads(body)
    if result.get("error"):
        raise ExecutionError(str(result["error"])[:500])
    decode_files(result.get("files", []), limit=MAX_OUTPUT_BYTES)
    if not isinstance(result.get("exit_code"), int):
        raise ExecutionError("执行服务返回了无效状态。")
    return result
