"""Keep a completed XR scene visible without advancing simulation or recording."""

from __future__ import annotations

import math
import time
from collections.abc import Callable


def _error_reason(operation: str, error: BaseException) -> str:
    try:
        detail = str(error)
    except BaseException:
        detail = ""
    return f"{operation}_error:{type(error).__name__}" + (f": {detail}" if detail else "")


def should_render_completed_scene(*, status: str, teleop: bool, args: object) -> bool:
    """Enable idle rendering only for a completed, explicitly opted-in VR trial."""
    return (
        status == "complete"
        and teleop
        and getattr(args, "teleop_device", "") == "vr"
        and getattr(args, "xr_keep_open_s", 0) > 0
    )


def render_completed_scene(
    duration_s: float,
    *,
    render: Callable[[], None],
    is_running: Callable[[], bool],
    stop_requested: Callable[[], bool],
    log: Callable[[str], None],
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Render the final scene until a wall-clock deadline or normal shutdown."""
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and nonnegative")
    if duration_s == 0:
        return "disabled"

    reason = "deadline"
    operation = "clock"
    try:
        deadline = clock() + duration_s
        operation = "enter_log"
        log(f"[AIRSEAI_XR_IDLE] entering seconds={duration_s:g}")
        while True:
            operation = "stop_requested"
            if stop_requested():
                reason = "stop_requested"
                break
            operation = "is_running"
            if not is_running():
                reason = "app_stopped"
                break
            operation = "clock"
            if clock() >= deadline:
                break
            operation = "render"
            render()
    except BaseException as error:
        # Display failures must not overwrite completed recording results.
        reason = _error_reason(operation, error)
    try:
        log(f"[AIRSEAI_XR_IDLE] exited reason={reason}")
    except BaseException as error:
        # Preserve the first error if logging also fails.
        if reason in ("deadline", "stop_requested", "app_stopped"):
            reason = _error_reason("exit_log", error)
    return reason
