"""Attribution for explicit CLI consent.

``--yes`` proves that a caller supplied an affirmative flag. It does not prove who
typed it. This module records the observable execution surface without upgrading a
subprocess invocation into a human act:

* interactive stdin TTY -> ``actor=human``
* non-interactive stdin -> ``actor=human_via_agent``
* unavailable telemetry -> ``actor=unknown``

Parent-process metadata is best-effort and intentionally excludes command-line
arguments, which may contain secrets. Attribution is evidence, not authentication.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

from . import config
from .util import append_jsonl, now_tz_iso


CONSENT_ACTORS = frozenset({"human", "human_via_agent", "unknown"})


def _windows_stdin_is_console(stream: Any) -> bool | None:
    """Distinguish a real Windows console from NUL and redirected char devices."""

    if os.name != "nt":
        return None
    try:
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_ulong()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - failure must not be upgraded to human
        return None


def _stdin_tty_state() -> bool | None:
    try:
        stream = sys.stdin
        probe = getattr(stream, "isatty", None)
        if not callable(probe):
            return None
        value = probe()
        if not isinstance(value, bool):
            return None
        if value is False:
            return False
        if os.name == "nt":
            # CPython may report NUL (including subprocess.DEVNULL) as a TTY.
            # Only GetConsoleMode confirms an actual Windows console handle.
            return _windows_stdin_is_console(stream)
        return True
    except Exception:  # noqa: BLE001 - attribution must stay honest, not crash a gate
        return None


def _windows_process_image(pid: int) -> str:
    if os.name != "nt":
        return ""
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ""
        try:
            size = ctypes.c_ulong(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return ""
            return str(buffer.value)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        return ""


def _proc_process_image(pid: int) -> str:
    proc_exe = Path("/proc") / str(pid) / "exe"
    proc_comm = Path("/proc") / str(pid) / "comm"
    try:
        if proc_exe.exists():
            return str(proc_exe.resolve())
    except OSError:
        pass
    try:
        if proc_comm.is_file():
            return proc_comm.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        pass
    return ""


def _parent_process() -> dict[str, Any]:
    try:
        pid = int(os.getppid())
    except Exception:  # noqa: BLE001
        return {"pid": None, "name": "unknown", "executable": "", "available": False}
    image = _windows_process_image(pid) or _proc_process_image(pid)
    return {
        "pid": pid,
        "name": Path(image).name if image else "unknown",
        "executable": image,
        "available": bool(image),
    }


def capture_consent_attribution() -> dict[str, Any]:
    """Return a JSON-safe snapshot of the observable consent surface."""

    tty_state = _stdin_tty_state()
    if tty_state is True:
        actor = "human"
        method = "interactive_tty"
    elif tty_state is False:
        actor = "human_via_agent"
        method = "non_interactive_or_subprocess"
    else:
        actor = "unknown"
        method = "unknown"
    return {
        "actor": actor,
        "method": method,
        "stdin_isatty": tty_state,
        "parent_process": _parent_process(),
        "recorded_at": now_tz_iso(),
        "attribution_is_authentication": False,
    }


def record_consent_event(
    action: str,
    target: str,
    attribution: dict[str, Any],
    *,
    outcome: str,
    detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one durable consent observation for success or refusal."""

    config.ensure_harness()
    event = {
        "schema": "companybrain.engineer.consent_event.v1",
        "action": str(action),
        "target": str(target),
        "outcome": str(outcome),
        "detail": str(detail)[:1200],
        "attribution": attribution,
        "actor": str(attribution.get("actor") or "unknown"),
        "recorded_at": now_tz_iso(),
        "metadata": metadata or {},
    }
    append_jsonl(config.ENGINEER_CONSENT_EVENTS_FILE, event)
    return event
