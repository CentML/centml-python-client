import asyncio
import json
import shutil
import signal
import sys
import termios
import tty
import urllib.parse

import pyte
import websockets

from centml.sdk import PodStatus

# exec_session wraps commands between BEGIN/END markers so it can separate
# real command output from shell noise (prompt, echoed input, login banners).
BEGIN_MARKER = "__CENTML_BEGIN_5f3a__"
END_MARKER = "__CENTML_END_5f3a__"

# printf octal \137 = underscore. The decoded output matches BEGIN/END_MARKER,
# but the literal command text does NOT, so shell echo won't trigger false matches.
PRINTF_BEGIN = BEGIN_MARKER.replace("__", r"\137\137")
PRINTF_END = END_MARKER.replace("__", r"\137\137")


def build_ws_url(api_url, deployment_id, pod_name, shell_type=None):
    parsed = urllib.parse.urlparse(api_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_base = parsed._replace(scheme=ws_scheme).geturl()
    url = f"{ws_base}/deployments/{deployment_id}/terminal?pod={urllib.parse.quote(pod_name)}"
    if shell_type:
        url += f"&shell={urllib.parse.quote(shell_type)}"
    return url


def get_running_pods(cclient, deployment_id) -> list[str]:
    status = cclient.get_status_v3(deployment_id)
    running_pods = []
    for revision in status.revision_pod_details_list or []:
        for pod in revision.pod_details_list or []:
            if pod.status == PodStatus.RUNNING and pod.name:
                running_pods.append(pod.name)
    return running_pods


async def forward_io(ws, term_size, shutdown):
    """Bidirectional forwarding between local stdin/stdout and WebSocket.

    Output is written directly to stdout (the local terminal is in raw
    mode, so ANSI sequences from the remote PTY are rendered natively).

    The platform API proxy sends a close frame (code=1000) when the
    remote shell exits, so _read_ws terminates via ConnectionClosed.

    Args:
        ws: WebSocket connection.
        term_size: Mutable list ``[cols, rows]`` kept up-to-date by the
            SIGWINCH handler in ``interactive_session``.
        shutdown: asyncio.Event set by signal handlers to request exit.

    Returns:
        The exit code (always 0 for interactive sessions).
    """
    loop = asyncio.get_running_loop()
    stdin_fd = sys.stdin.fileno()
    stdin_closed = asyncio.Event()

    async def _read_ws():
        try:
            while True:
                raw_msg = await ws.recv()
                msg = json.loads(raw_msg)
                data = msg.get("data", "")
                if data:
                    sys.stdout.buffer.write(data.encode("utf-8", errors="replace"))
                    sys.stdout.buffer.flush()
                elif msg.get("error"):
                    sys.stderr.buffer.write(f"Error: {msg['error']}\r\n".encode())
                    sys.stderr.buffer.flush()
        except websockets.ConnectionClosed:
            return

    async def _read_stdin():
        read_queue = asyncio.Queue()

        def _on_stdin_ready():
            data = sys.stdin.buffer.read1(4096)
            if data:
                read_queue.put_nowait(data)
            else:
                stdin_closed.set()

        loop.add_reader(stdin_fd, _on_stdin_ready)
        try:
            while not stdin_closed.is_set() and not shutdown.is_set():
                try:
                    data = await asyncio.wait_for(read_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "operation": "stdin",
                                "data": data.decode("utf-8", errors="replace"),
                                "rows": term_size[1],
                                "cols": term_size[0],
                            }
                        )
                    )
                except websockets.ConnectionClosed:
                    return
        finally:
            loop.remove_reader(stdin_fd)

    async def _watch_shutdown():
        while not shutdown.is_set():
            await asyncio.sleep(0.2)

    task_ws = asyncio.create_task(_read_ws())
    task_stdin = asyncio.create_task(_read_stdin())
    task_shutdown = asyncio.create_task(_watch_shutdown())
    tasks = [task_ws, task_stdin, task_shutdown]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for t in pending:
        t.cancel()
    for t in pending:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    for t in done:
        if t.exception() is not None:
            raise t.exception()
    return 0


async def interactive_session(ws_url, token):
    """Run an interactive terminal session over WebSocket.

    Enters raw mode, forwards I/O bidirectionally, and restores terminal
    on exit.  SIGTERM and SIGHUP are caught to ensure terminal settings
    are always restored.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    loop = asyncio.get_running_loop()
    signals_installed = False
    try:
        tty.setraw(fd)
        cols, rows = shutil.get_terminal_size()
        term_size = [cols, rows]

        shutdown = asyncio.Event()
        loop.add_signal_handler(signal.SIGTERM, shutdown.set)
        loop.add_signal_handler(signal.SIGHUP, shutdown.set)
        signals_installed = True

        headers = {"Authorization": f"Bearer {token}"}
        async with websockets.connect(ws_url, additional_headers=headers, close_timeout=2) as ws:

            def _send_resize():
                c, r = shutil.get_terminal_size()
                term_size[0], term_size[1] = c, r
                asyncio.ensure_future(ws.send(json.dumps({"operation": "resize", "rows": r, "cols": c})))

            loop.add_signal_handler(signal.SIGWINCH, _send_resize)

            await ws.send(json.dumps({"operation": "resize", "rows": rows, "cols": cols}))
            try:
                exit_code = await forward_io(ws, term_size, shutdown)
            finally:
                loop.remove_signal_handler(signal.SIGWINCH)

            return exit_code
    finally:
        if signals_installed:
            loop.remove_signal_handler(signal.SIGTERM)
            loop.remove_signal_handler(signal.SIGHUP)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _pyte_extract_text(line_stream, line_screen, text):
    """Feed text through a single-row pyte screen and return visible characters.

    More robust than regex ANSI stripping: pyte interprets all VT100/VT220
    sequences including OSC, cursor repositioning, and truecolor escapes.
    """
    line_screen.reset()
    line_stream.feed(text)
    return "".join(line_screen.buffer[0][col].data for col in range(line_screen.columns)).rstrip()


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
        try:
            async for raw_msg in ws:
                msg = json.loads(raw_msg)
                if msg.get("data"):
                    buffer += msg["data"]
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        clean = _pyte_extract_text(line_stream, line_screen, line.rstrip("\r"))
                        if BEGIN_MARKER in clean:
                            is_capturing = True
                            continue
                        if END_MARKER in clean:
                            parts = clean.split(END_MARKER + ":")
                            if len(parts) > 1:
                                try:
                                    exit_code = int(parts[1].strip())
                                except ValueError:
                                    pass
                            is_done = True
                            break
                        if is_capturing:
                            sys.stdout.write(line + "\n")
                            sys.stdout.flush()
                elif msg.get("error"):
                    sys.stderr.write(f"Error: {msg['error']}\n")
                    return 1
                if is_done:
                    break
        except websockets.ConnectionClosed:
            pass
        return exit_code
