"""WebSocket session logic for shell and exec commands (no Click dependency)."""

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import termios
import tty
import urllib.parse
from typing import Optional, Tuple

import pyte
import websockets

from centml.sdk import PodStatus
from centml.sdk.shell.exceptions import NoPodAvailableError, PodNotFoundError
from centml.sdk.shell.renderer import pyte_extract_text, render_dirty

_log = logging.getLogger("centml.sdk.shell")

BEGIN_MARKER = "__CENTML_BEGIN_5f3a__"
END_MARKER = "__CENTML_END_5f3a__"

# printf octal \137 = underscore. The decoded output matches BEGIN/END_MARKER,
# but the literal command text does NOT, so shell echo won't trigger false matches.
PRINTF_BEGIN = r"\137\137CENTML_BEGIN_5f3a\137\137"
PRINTF_END = r"\137\137CENTML_END_5f3a\137\137"


def setup_debug_log():
    """Configure file-based debug logging (stdout unusable in raw mode)."""
    log_path = os.environ.get("CENTML_SHELL_DEBUG_LOG", "/tmp/centml_shell_debug.log")
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S"))
    _log.addHandler(handler)
    _log.setLevel(logging.DEBUG)
    _log.debug("=== shell debug log started (pid=%d) ===", os.getpid())


def build_ws_url(api_url, deployment_id, pod_name, shell_type=None):
    """Build the WebSocket URL for a terminal connection."""
    parsed = urllib.parse.urlparse(api_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_base = parsed._replace(scheme=ws_scheme).geturl()
    url = f"{ws_base}/deployments/{deployment_id}/terminal?pod={urllib.parse.quote(pod_name)}"
    if shell_type:
        url += f"&shell={urllib.parse.quote(shell_type)}"
    return url


def resolve_pod(cclient, deployment_id, pod_name=None) -> Tuple[str, Optional[str]]:
    """Resolve which pod to connect to.

    Args:
        cclient: CentMLClient instance.
        deployment_id: The deployment ID.
        pod_name: Optional specific pod name to target.

    Returns:
        Tuple of (pod_name, optional_warning_message).

    Raises:
        NoPodAvailableError: If no running pods found.
        PodNotFoundError: If specified pod not found among running pods.
    """
    status = cclient.get_status_v3(deployment_id)
    running_pods = []
    for revision in status.revision_pod_details_list or []:
        for pod in revision.pod_details_list or []:
            if pod.status == PodStatus.RUNNING and pod.name:
                running_pods.append(pod.name)

    if not running_pods:
        raise NoPodAvailableError(f"No running pods found for deployment {deployment_id}")

    if pod_name is not None:
        if pod_name not in running_pods:
            pods_list = ", ".join(running_pods)
            raise PodNotFoundError(f"Pod '{pod_name}' not found. Available running pods: {pods_list}")
        return pod_name, None

    warning = None
    if len(running_pods) > 1:
        warning = (
            f"Multiple running pods found, connecting to {running_pods[0]}. " f"Use --pod to specify a different pod."
        )
    return running_pods[0], warning


async def forward_io(ws, screen, stream, shutdown):
    """Bidirectional forwarding between local stdin/stdout and WebSocket.

    Output flows through a pyte terminal emulator so that cursor
    addressing, line wrapping, and colors are rendered correctly
    regardless of the remote PTY dimensions.

    The platform API proxy sends a close frame (code=1000) when the
    remote shell exits, so _read_ws terminates via ConnectionClosed.

    Args:
        ws: WebSocket connection.
        screen: pyte.Screen instance sized to the local terminal.
        stream: pyte.Stream attached to *screen*.
        shutdown: asyncio.Event set by signal handlers to request exit.

    Returns:
        The exit code (always 0 for interactive sessions).
    """
    loop = asyncio.get_running_loop()
    stdin_fd = sys.stdin.fileno()
    stdin_closed = asyncio.Event()

    async def _read_ws():
        _log.debug("[read_ws] started")
        msg_count = 0
        try:
            while True:
                raw_msg = await ws.recv()
                msg_count += 1
                msg = json.loads(raw_msg)
                keys = list(msg.keys())
                data = msg.get("data", "")
                data_snippet = repr(data[:120]) if data else ""
                _log.debug("[read_ws] msg#%d keys=%s data=%s", msg_count, keys, data_snippet)
                if data:
                    stream.feed(data.replace("\n", "\r\n"))
                    render_dirty(screen, sys.stdout.buffer)
                elif msg.get("error"):
                    _log.debug("[read_ws] error: %s", msg["error"])
                    stream.feed(f"Error: {msg['error']}\r\n")
                    render_dirty(screen, sys.stdout.buffer)
        except websockets.ConnectionClosed as exc:
            _log.debug("[read_ws] ConnectionClosed after %d msgs: %s", msg_count, exc)
            return

    async def _read_stdin():
        _log.debug("[read_stdin] started")
        read_queue = asyncio.Queue()

        def _on_stdin_ready():
            data = sys.stdin.buffer.read1(4096)
            if data:
                read_queue.put_nowait(data)
            else:
                _log.debug("[read_stdin] stdin EOF")
                stdin_closed.set()

        loop.add_reader(stdin_fd, _on_stdin_ready)
        try:
            while not stdin_closed.is_set() and not shutdown.is_set():
                try:
                    data = await asyncio.wait_for(read_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                _log.debug("[read_stdin] sending %d bytes: %s", len(data), repr(data[:40]))
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "operation": "stdin",
                                "data": data.decode("utf-8", errors="replace"),
                                "rows": screen.lines,
                                "cols": screen.columns,
                            }
                        )
                    )
                except websockets.ConnectionClosed:
                    _log.debug("[read_stdin] ws closed on send")
                    return
            _log.debug(
                "[read_stdin] loop exited: stdin_closed=%s shutdown=%s", stdin_closed.is_set(), shutdown.is_set()
            )
        finally:
            loop.remove_reader(stdin_fd)

    async def _watch_shutdown():
        while not shutdown.is_set():
            await asyncio.sleep(0.2)

    _log.debug("[forward_io] creating tasks")
    task_ws = asyncio.create_task(_read_ws())
    task_stdin = asyncio.create_task(_read_stdin())
    task_shutdown = asyncio.create_task(_watch_shutdown())
    tasks = [task_ws, task_stdin, task_shutdown]
    task_names = {id(task_ws): "read_ws", id(task_stdin): "read_stdin", id(task_shutdown): "watch_shutdown"}

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    done_names = [task_names[id(t)] for t in done]
    pending_names = [task_names[id(t)] for t in pending]
    _log.debug("[forward_io] first completed: done=%s pending=%s", done_names, pending_names)

    for t in pending:
        t.cancel()
    for t in pending:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    for t in done:
        if t.exception() is not None:
            _log.debug("[forward_io] task exception: %s", t.exception())
            raise t.exception()
    _log.debug("[forward_io] returning exit_code=0")
    return 0


async def interactive_session(ws_url, token):
    """Run an interactive terminal session over WebSocket.

    Enters raw mode, forwards I/O bidirectionally, and restores terminal
    on exit.  SIGTERM and SIGHUP are caught to ensure terminal settings
    are always restored.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        cols, rows = shutil.get_terminal_size()

        screen = pyte.Screen(cols, rows)
        stream = pyte.Stream(screen)

        # Switch to alternate screen buffer (disables scrollback) and clear.
        sys.stdout.buffer.write(b"\033[?1049h\033[2J\033[H")
        sys.stdout.buffer.flush()

        loop = asyncio.get_running_loop()

        shutdown = asyncio.Event()
        loop.add_signal_handler(signal.SIGTERM, shutdown.set)
        loop.add_signal_handler(signal.SIGHUP, shutdown.set)

        headers = {"Authorization": f"Bearer {token}"}
        _log.debug("[session] connecting to %s", ws_url)
        async with websockets.connect(ws_url, additional_headers=headers, close_timeout=2) as ws:
            _log.debug("[session] connected, sending resize %dx%d", cols, rows)

            def _send_resize():
                c, r = shutil.get_terminal_size()
                screen.resize(r, c)
                screen.dirty.update(range(r))
                asyncio.ensure_future(ws.send(json.dumps({"operation": "resize", "rows": r, "cols": c})))

            loop.add_signal_handler(signal.SIGWINCH, _send_resize)

            await ws.send(json.dumps({"operation": "resize", "rows": rows, "cols": cols}))
            try:
                exit_code = await forward_io(ws, screen, stream, shutdown)
            finally:
                loop.remove_signal_handler(signal.SIGWINCH)

            _log.debug("[session] exiting with code %d", exit_code)
            return exit_code
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        loop.remove_signal_handler(signal.SIGHUP)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Leave alternate screen buffer, restore cursor and attributes.
        sys.stdout.buffer.write(b"\033[?1049l\033[?25h\033[0m")
        sys.stdout.buffer.flush()


async def exec_session(ws_url, token, command):
    """Execute a command in a pod and return its exit code.

    Does not enter raw mode -- output is pipe-friendly.
    Suppresses shell echo and uses markers to capture only command output.
    """
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    # Single-row screen for interpreting escape sequences in marker detection.
    line_screen = pyte.Screen(cols, 1)
    line_stream = pyte.Stream(line_screen)
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(ws_url, additional_headers=headers, close_timeout=2) as ws:
        await ws.send(json.dumps({"operation": "resize", "rows": rows, "cols": cols}))

        # Suppress echo/bracketed-paste, emit begin marker, run command,
        # emit end marker with exit code, then exit.
        # Markers use printf octal escapes so the literal marker string
        # doesn't appear in the command echo.
        wrapped = (
            f"stty -echo 2>/dev/null; printf '\\033[?2004l';"
            f" printf '{PRINTF_BEGIN}\\n';"
            f" {command};"
            f" __ec=$?;"
            f" printf '\\n{PRINTF_END}:%d\\n' \"$__ec\";"
            f" exit $__ec\n"
        )

        await ws.send(json.dumps({"operation": "stdin", "data": wrapped}))

        exit_code = 0
        buffer = ""
        is_capturing = False
        is_done = False
        msg_count = 0
        try:
            async for raw_msg in ws:
                msg_count += 1
                msg = json.loads(raw_msg)
                keys = list(msg.keys())
                _log.debug("[exec] msg#%d keys=%s data_len=%d", msg_count, keys, len(msg.get("data", "")))
                if msg.get("data"):
                    buffer += msg["data"]
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        clean = pyte_extract_text(line_stream, line_screen, line.rstrip("\r"))
                        if BEGIN_MARKER in clean:
                            _log.debug("[exec] BEGIN marker found")
                            is_capturing = True
                            continue
                        if END_MARKER in clean:
                            parts = clean.split(END_MARKER + ":")
                            if len(parts) > 1:
                                try:
                                    exit_code = int(parts[1].strip())
                                except ValueError:
                                    pass
                            _log.debug("[exec] END marker, exit_code=%d", exit_code)
                            is_done = True
                            break
                        if is_capturing:
                            sys.stdout.write(line + "\n")
                            sys.stdout.flush()
                elif msg.get("error"):
                    _log.debug("[exec] error: %s", msg["error"])
                    sys.stderr.write(f"Error: {msg['error']}\n")
                    return 1
                if is_done:
                    _log.debug("[exec] done, breaking")
                    break
        except websockets.ConnectionClosed as exc:
            _log.debug("[exec] ConnectionClosed: %s", exc)
        _log.debug("[exec] returning exit_code=%d", exit_code)
        return exit_code
