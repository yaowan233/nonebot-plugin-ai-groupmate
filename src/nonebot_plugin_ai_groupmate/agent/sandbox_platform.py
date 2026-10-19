"""OS-specific write confinement. Missing runners always fail closed."""

import os
import sys
import shutil
from pathlib import Path

from .sandbox_launcher import supported_platform as supports_landlock


class SandboxUnavailable(ValueError):
    pass


def supported_platform() -> bool:
    return sys.platform in {"win32", "darwin"} or supports_landlock()


def windows_runner(node: str, configured: str = "") -> Path:
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise SandboxUnavailable("Windows 沙箱 runner 配置必须是存在的绝对文件路径；未降级为 local。")
        return path.resolve()
    # Only administrator-installed runtime locations, never the task cwd.
    roots = [Path(node).parent / "node_modules", Path(node).resolve().parent / "node_modules"]
    if os.environ.get("APPDATA"):
        roots.append(Path(os.environ["APPDATA"]) / "npm" / "node_modules")
    for root in roots:
        for package_root in (root, root / "@deepseek-ai" / "dsh" / "node_modules"):
            path = package_root / "@deepseek-ai" / "dsh-sandbox-windows-acl" / "lib" / "runner.js"
            if path.is_file():
                return path.resolve()
    raise SandboxUnavailable("未找到 Windows ACL 沙箱运行器；请安装 @deepseek-ai/dsh-sandbox-windows-acl 或配置 runner 绝对路径。未降级为 local。")


def seatbelt_profile(workspace: Path) -> str:
    # SBPL literals: escape path characters, never interpolate shell syntax.
    root = str(workspace.resolve(strict=True)).replace("\\", "\\\\").replace('"', '\\"')
    if any(ord(char) < 32 for char in root):
        raise SandboxUnavailable("沙箱路径不能包含控制字符。")
    return f'(version 1) (allow default) (deny file-write*) (allow file-write* (literal "/dev/null") (subpath "{root}"))'


def sandbox_command(argv: list[str], workspace: Path, *, windows_temp: Path | None = None, runner: str = "") -> list[str]:
    if not supported_platform():
        raise SandboxUnavailable("当前平台不支持 sandbox；未降级为 local。")
    if sys.platform == "linux":
        launcher = Path(__file__).with_name("sandbox_launcher.py").resolve()
        return [sys.executable, "-I", "-S", str(launcher), str(workspace), "--", *argv]
    if sys.platform == "darwin":
        executable = Path("/usr/bin/sandbox-exec")
        if not executable.is_file():
            raise SandboxUnavailable("未找到 macOS sandbox-exec；未降级为 local。")
        return [str(executable), "-p", seatbelt_profile(workspace), *argv]
    node = shutil.which("node")
    if not node:
        raise SandboxUnavailable("Windows 沙箱需要 Node.js；未降级为 local。")
    if windows_temp is None:
        raise SandboxUnavailable("Windows 沙箱缺少私有临时区。")
    return [node, str(windows_runner(node, runner)), "--workspace", str(workspace), "--temp", str(windows_temp), "--mode", "workspace-write", "--", *argv]
