from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("sandbox wrapper: missing command", file=sys.stderr)
        return 2

    allowed_root = os.environ.get("ENGINEGEN_SANDBOX_ALLOWED_ROOT")
    extra_roots = os.environ.get("ENGINEGEN_SANDBOX_EXTRA_ROOTS", "")
    fs_mode = os.environ.get("ENGINEGEN_SANDBOX_FS_MODE", "auto").lower()
    require_fs = os.environ.get("ENGINEGEN_SANDBOX_REQUIRE_FS", "1") == "1"
    if allowed_root:
        root = Path(allowed_root)
        cwd = Path.cwd()
        if not _is_within(cwd, root):
            print(f"sandbox wrapper: cwd {cwd} not within {root}", file=sys.stderr)
            return 3
    else:
        root = Path.cwd()

    policy = os.environ.get("ENGINEGEN_SANDBOX_POLICY", "strict").lower()
    deny_network = os.environ.get("ENGINEGEN_SANDBOX_DENY_NETWORK", "1") == "1"
    if deny_network:
        unshare = shutil.which("unshare")
        if not unshare:
            print("sandbox wrapper: unshare not available for network isolation", file=sys.stderr)
            return 4
        args = [unshare, "--user", "--map-root-user", "--net", "--"] + args

    args = _wrap_filesystem(args, root, extra_roots, fs_mode, require_fs)
    if args is None:
        return 5

    os.execvpe(args[0], args, os.environ)
    return 1


def _wrap_filesystem(
    args: list[str],
    root: Path,
    extra_roots: str,
    fs_mode: str,
    require_fs: bool,
) -> list[str] | None:
    if fs_mode == "none":
        if require_fs:
            print("sandbox wrapper: filesystem isolation required but disabled", file=sys.stderr)
            return None
        return args

    bwrap = shutil.which("bwrap")
    if bwrap is None and fs_mode != "none":
        if require_fs:
            print("sandbox wrapper: bwrap not available for filesystem isolation", file=sys.stderr)
            return None
        return args

    if bwrap is None:
        return args

    binds: list[str] = [
        bwrap,
        "--die-with-parent",
        "--unshare-all",
    ]

    if os.environ.get("ENGINEGEN_SANDBOX_DENY_NETWORK", "1") != "1":
        binds.append("--share-net")

    for path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(path).exists():
            binds.extend(["--ro-bind", path, path])
    binds.extend(["--dev", "/dev"])
    binds.extend(["--proc", "/proc"])
    binds.extend(["--tmpfs", "/tmp"])

    binds.extend(["--bind", str(root), "/work"])
    binds.extend(["--chdir", "/work"])

    if extra_roots:
        for entry in extra_roots.split(":"):
            entry = entry.strip()
            if not entry:
                continue
            path = Path(entry)
            if not path.exists():
                continue
            binds.extend(["--ro-bind", str(path), str(path)])

    binds.append("--")
    return binds + args


if __name__ == "__main__":
    raise SystemExit(main())
