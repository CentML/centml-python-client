"""SDK shell module -- reusable shell/exec session logic (no Click dependency)."""

from centml.sdk.shell.exceptions import NoPodAvailableError, PodNotFoundError, ShellError
from centml.sdk.shell.renderer import char_to_sgr, color_sgr, pyte_extract_text, render_dirty
from centml.sdk.shell.session import (
    BEGIN_MARKER,
    END_MARKER,
    PRINTF_BEGIN,
    PRINTF_END,
    build_ws_url,
    exec_session,
    forward_io,
    interactive_session,
    resolve_pod,
)

__all__ = [
    "ShellError",
    "NoPodAvailableError",
    "PodNotFoundError",
    "color_sgr",
    "char_to_sgr",
    "render_dirty",
    "pyte_extract_text",
    "build_ws_url",
    "resolve_pod",
    "forward_io",
    "interactive_session",
    "exec_session",
    "BEGIN_MARKER",
    "END_MARKER",
    "PRINTF_BEGIN",
    "PRINTF_END",
]
