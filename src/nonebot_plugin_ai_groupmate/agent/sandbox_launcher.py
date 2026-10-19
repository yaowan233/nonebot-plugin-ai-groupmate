"""Single-threaded Linux Landlock launcher; never import this into a worker to restrict it.

Own implementation of the Linux UAPI, not a copy of DSH. Like DSH's file-effect
policy, this restricts writes, not reads or networking. Requires ABI >= 3 so
truncate and cross-directory rename/link are covered. No fallback execution.
Reference: https://docs.kernel.org/userspace-api/landlock.html
"""

import os
import sys
import ctypes
import platform
from pathlib import Path

# Linux x86_64 and aarch64 share these syscall numbers.
CREATE_RULESET, ADD_RULE, RESTRICT_SELF = 444, 445, 446
WRITE_FILE = 1 << 1
TRUNCATE = 1 << 14
WRITE_RIGHTS = WRITE_FILE | sum(1 << bit for bit in range(4, 15))


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def supported_platform() -> bool:
    return sys.platform == "linux" and platform.machine().lower() in {"x86_64", "aarch64"} and ctypes.sizeof(ctypes.c_void_p) == 8


def restrict_writes(workspace: Path) -> None:
    if not supported_platform():
        raise RuntimeError("sandbox requires Linux x86_64/aarch64 with Landlock ABI >= 3")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long

    def checked(value: int, operation: str) -> int:
        if value < 0:
            number = ctypes.get_errno()
            raise OSError(number, operation + ": " + os.strerror(number))
        return value

    abi = checked(libc.syscall(CREATE_RULESET, 0, 0, 1), "Landlock ABI probe")
    if abi < 3:
        raise RuntimeError(f"Landlock ABI {abi} is insufficient; ABI >= 3 required")
    root = workspace.resolve(strict=True)
    if not root.is_dir() or root == Path("/"):
        raise ValueError("sandbox workspace must be a dedicated directory")
    attr = Ruleset(WRITE_RIGHTS)
    ruleset = checked(libc.syscall(CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0), "create ruleset")
    try:
        for path, rights in ((root, WRITE_RIGHTS), (Path("/dev/null"), WRITE_FILE | TRUNCATE)):
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, descriptor)
                checked(libc.syscall(ADD_RULE, ruleset, 1, ctypes.byref(rule), 0), "add path rule")
            finally:
                os.close(descriptor)
        checked(libc.prctl(38, 1, 0, 0, 0), "set no_new_privs")
        checked(libc.syscall(RESTRICT_SELF, ruleset, 0), "restrict self")
    finally:
        os.close(ruleset)


def main() -> int:
    try:
        if len(sys.argv) < 4 or sys.argv[2] != "--":
            raise ValueError("usage: sandbox_launcher.py WORKSPACE -- COMMAND [ARGS...]")
        restrict_writes(Path(sys.argv[1]))
        os.execv(sys.argv[3], sys.argv[3:])
    except Exception as error:
        print(f"Sandbox launch refused (no fallback): {error}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    sys.exit(main())
