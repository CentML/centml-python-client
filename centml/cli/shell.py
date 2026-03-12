"""CLI commands for interactive shell and command execution in deployment pods."""

import asyncio
import json
import shutil
import signal
import sys
import termios
import tty
import urllib.parse

import click
import pyte
import websockets

from centml.cli.cluster import handle_exception
from centml.sdk import PodStatus, auth
from centml.sdk.api import get_centml_client
from centml.sdk.config import settings


# ---------------------------------------------------------------------------
# pyte screen renderer -- converts pyte's in-memory screen buffer to ANSI
# escape sequences for the local terminal.
# ---------------------------------------------------------------------------

_PYTE_FG_TO_SGR = {
    "default": "39",
    "black": "30",
    "red": "31",
    "green": "32",
    "brown": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "brightblack": "90",
    "brightred": "91",
    "brightgreen": "92",
    "brightbrown": "93",
    "brightblue": "94",
    "brightmagenta": "95",
    "brightcyan": "96",
    "brightwhite": "97",
}

_PYTE_BG_TO_SGR = {
    "default": "49",
    "black": "40",
    "red": "41",
    "green": "42",
    "brown": "43",
    "blue": "44",
    "magenta": "45",
    "cyan": "46",
    "white": "47",
    "brightblack": "100",
    "brightred": "101",
    "brightgreen": "102",
    "brightbrown": "103",
    "brightblue": "104",
    "brightmagenta": "105",
    "brightcyan": "106",
    "brightwhite": "107",
}


def _color_sgr(color, is_bg=False):
    """Convert a pyte color value to an SGR parameter string."""
    table = _PYTE_BG_TO_SGR if is_bg else _PYTE_FG_TO_SGR
    if color in table:
        default_val = "49" if is_bg else "39"
        code = table[color]
        return code if code != default_val else ""
    # 6-char hex -> truecolor
    if len(color) == 6:
        try:
            r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
            prefix = "48" if is_bg else "38"
            return f"{prefix};2;{r};{g};{b}"
        except ValueError:
            return ""
    return ""


def _char_to_sgr(char):
    """Build the ANSI SGR parameter string for a pyte Char's attributes."""
    parts = []
    if char.bold:
        parts.append("1")
    if char.italics:
        parts.append("3")
    if char.underscore:
        parts.append("4")
    if char.blink:
        parts.append("5")
    if char.reverse:
        parts.append("7")
    if char.strikethrough:
        parts.append("9")
    fg = _color_sgr(char.fg, is_bg=False)
    if fg:
        parts.append(fg)
    bg = _color_sgr(char.bg, is_bg=True)
    if bg:
        parts.append(bg)
    return ";".join(parts)


def _render_dirty(screen, output):
    """Render only the dirty lines from the pyte Screen to the terminal.

    Args:
        screen: pyte.Screen instance.
        output: Writable binary stream (e.g. sys.stdout.buffer).
    """
    parts = []
    for row in sorted(screen.dirty):
        # Position cursor at row (1-based), column 1; clear line.
        parts.append(f"\033[{row + 1};1H\033[2K")
        prev_sgr = ""
        line_chars = []
        for col in range(screen.columns):
            char = screen.buffer[row][col]
            if char.data == "":
                continue
            sgr = _char_to_sgr(char)
            if sgr != prev_sgr:
                line_chars.append(f"\033[0m\033[{sgr}m" if sgr else "\033[0m")
                prev_sgr = sgr
            line_chars.append(char.data)
        text = "".join(line_chars).rstrip()
        parts.append(text)
    # Reset attributes, position cursor.
    parts.append("\033[0m")
    parts.append(f"\033[{screen.cursor.y + 1};{screen.cursor.x + 1}H")
    if screen.cursor.hidden:
        parts.append("\033[?25l")
    else:
        parts.append("\033[?25h")
    screen.dirty.clear()
    output.write("".join(parts).encode("utf-8"))
    output.flush()


def _build_ws_url(api_url, deployment_id, pod_name, shell_type=None):
    """Build the WebSocket URL for a terminal connection."""
    parsed = urllib.parse.urlparse(api_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_base = parsed._replace(scheme=ws_scheme).geturl()
    url = f"{ws_base}/deployments/{deployment_id}/terminal?pod={urllib.parse.quote(pod_name)}"
    if shell_type:
        url += f"&shell={urllib.parse.quote(shell_type)}"
    return url


def _resolve_pod(cclient, deployment_id, pod_name=None):
    """Resolve which pod to connect to.

    Args:
        cclient: CentMLClient instance.
        deployment_id: The deployment ID.
        pod_name: Optional specific pod name to target.

    Returns:
        The pod name string to connect to.

    Raises:
        click.ClickException: If no running pods or specified pod not found.
    """
    status = cclient.get_status_v3(deployment_id)
    running_pods = []
    for revision in status.revision_pod_details_list or []:
        for pod in revision.pod_details_list or []:
            if pod.status == PodStatus.RUNNING and pod.name:
                running_pods.append(pod.name)

    if not running_pods:
        raise click.ClickException(
            f"No running pods found for deployment {deployment_id}"
        )

    if pod_name is not None:
        if pod_name not in running_pods:
            pods_list = ", ".join(running_pods)
            raise click.ClickException(
                f"Pod '{pod_name}' not found. Available running pods: {pods_list}"
            )
        return pod_name

    if len(running_pods) > 1:
        click.echo(
            f"Multiple running pods found, connecting to {running_pods[0]}. "
            f"Use --pod to specify a different pod.",
            err=True,
        )
    return running_pods[0]


async def _forward_io(ws, screen, stream):
    """Bidirectional forwarding between local stdin/stdout and WebSocket.

    Output flows through a pyte terminal emulator so that cursor
    addressing, line wrapping, and colors are rendered correctly
    regardless of the remote PTY dimensions.

    Args:
        ws: WebSocket connection.
        screen: pyte.Screen instance sized to the local terminal.
        stream: pyte.Stream attached to *screen*.

    Returns the remote exit code.
    """
    loop = asyncio.get_running_loop()
    exit_code = 0
    stdin_fd = sys.stdin.fileno()
    stdin_closed = asyncio.Event()

    async def _read_ws():
        nonlocal exit_code
        try:
            async for raw_msg in ws:
                msg = json.loads(raw_msg)
                if msg.get("data"):
                    # pyte expects \r\n; remote PTY may send bare \n
                    # (same as xterm.js ``convertEol: true``).
                    stream.feed(msg["data"].replace("\n", "\r\n"))
                    _render_dirty(screen, sys.stdout.buffer)
                elif msg.get("error"):
                    stream.feed(f"Error: {msg['error']}\r\n")
                    _render_dirty(screen, sys.stdout.buffer)
                if "Code" in msg:
                    exit_code = msg["Code"]
                    return
        except websockets.ConnectionClosed:
            # Backend proxy may not send a clean close frame when
            # ArgoCD disconnects after the remote shell exits.
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
            while not stdin_closed.is_set():
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
                                "rows": screen.lines,
                                "cols": screen.columns,
                            }
                        )
                    )
                except websockets.ConnectionClosed:
                    return
        finally:
            loop.remove_reader(stdin_fd)

    tasks = [asyncio.create_task(_read_ws()), asyncio.create_task(_read_stdin())]
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
    return exit_code


async def _interactive_session(ws_url, token):
    """Run an interactive terminal session over WebSocket.

    Enters raw mode, forwards I/O bidirectionally, and restores terminal on exit.
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

        headers = {"Authorization": f"Bearer {token}"}
        async with websockets.connect(
            ws_url, additional_headers=headers, close_timeout=2
        ) as ws:
            await ws.send(
                json.dumps({"operation": "resize", "rows": rows, "cols": cols})
            )

            loop = asyncio.get_running_loop()

            def _send_resize():
                c, r = shutil.get_terminal_size()
                screen.resize(r, c)
                screen.dirty.update(range(r))
                asyncio.ensure_future(
                    ws.send(json.dumps({"operation": "resize", "rows": r, "cols": c}))
                )

            loop.add_signal_handler(signal.SIGWINCH, _send_resize)

            try:
                exit_code = await _forward_io(ws, screen, stream)
            finally:
                loop.remove_signal_handler(signal.SIGWINCH)

            return exit_code
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Leave alternate screen buffer, restore cursor and attributes.
        sys.stdout.buffer.write(b"\033[?1049l\033[?25h\033[0m")
        sys.stdout.buffer.flush()


_BEGIN_MARKER = "__CENTML_BEGIN_5f3a__"
_END_MARKER = "__CENTML_END_5f3a__"

# printf octal \137 = underscore. The decoded output matches _BEGIN/_END_MARKER,
# but the literal command text does NOT, so shell echo won't trigger false matches.
_PRINTF_BEGIN = r"\137\137CENTML_BEGIN_5f3a\137\137"
_PRINTF_END = r"\137\137CENTML_END_5f3a\137\137"


def _pyte_extract_text(line_stream, line_screen, text):
    """Feed text through a single-row pyte screen and return visible characters.

    More robust than regex ANSI stripping: pyte interprets all VT100/VT220
    sequences including OSC, cursor repositioning, and truecolor escapes.
    """
    line_screen.reset()
    line_stream.feed(text)
    return "".join(
        line_screen.buffer[0][col].data for col in range(line_screen.columns)
    ).rstrip()


async def _exec_session(ws_url, token, command):
    """Execute a command in a pod and return its exit code.

    Does not enter raw mode -- output is pipe-friendly.
    Suppresses shell echo and uses markers to capture only command output.
    """
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    # Single-row screen for interpreting escape sequences in marker detection.
    line_screen = pyte.Screen(cols, 1)
    line_stream = pyte.Stream(line_screen)
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(
        ws_url, additional_headers=headers, close_timeout=2
    ) as ws:
        await ws.send(json.dumps({"operation": "resize", "rows": rows, "cols": cols}))

        # Suppress echo/bracketed-paste, emit begin marker, run command,
        # emit end marker with exit code, then exit.
        # Markers use printf octal escapes so the literal marker string
        # doesn't appear in the command echo.
        wrapped = (
            f"stty -echo 2>/dev/null; printf '\\033[?2004l';"
            f" printf '{_PRINTF_BEGIN}\\n';"
            f" {command};"
            f" __ec=$?;"
            f" printf '\\n{_PRINTF_END}:%d\\n' \"$__ec\";"
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
                        clean = _pyte_extract_text(
                            line_stream, line_screen, line.rstrip("\r")
                        )
                        if _BEGIN_MARKER in clean:
                            is_capturing = True
                            continue
                        if _END_MARKER in clean:
                            parts = clean.split(_END_MARKER + ":")
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
                if is_done or "Code" in msg:
                    if "Code" in msg:
                        exit_code = msg["Code"]
                    break
        except websockets.ConnectionClosed:
            pass
        return exit_code


@click.command(help="Open an interactive shell to a deployment pod")
@click.argument("deployment_id", type=int)
@click.option(
    "--pod", default=None, help="Specific pod name (auto-selects first running pod)"
)
@click.option(
    "--shell",
    "shell_type",
    default=None,
    type=click.Choice(["bash", "sh", "zsh"]),
    help="Shell type",
)
@handle_exception
def shell(deployment_id, pod, shell_type):
    if not sys.stdin.isatty():
        raise click.ClickException("Interactive shell requires a terminal (TTY)")

    with get_centml_client() as cclient:
        pod_name = _resolve_pod(cclient, deployment_id, pod)

    ws_url = _build_ws_url(
        settings.CENTML_PLATFORM_API_URL, deployment_id, pod_name, shell_type
    )
    token = auth.get_centml_token()
    exit_code = asyncio.run(_interactive_session(ws_url, token))
    sys.exit(exit_code)


@click.command(
    help="Execute a command in a deployment pod",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("deployment_id", type=int)
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option("--pod", default=None, help="Specific pod name")
@click.option(
    "--shell",
    "shell_type",
    default=None,
    type=click.Choice(["bash", "sh", "zsh"]),
    help="Shell type",
)
@handle_exception
def exec_cmd(deployment_id, command, pod, shell_type):
    with get_centml_client() as cclient:
        pod_name = _resolve_pod(cclient, deployment_id, pod)

    ws_url = _build_ws_url(
        settings.CENTML_PLATFORM_API_URL, deployment_id, pod_name, shell_type
    )
    token = auth.get_centml_token()
    cmd_str = " ".join(command)
    exit_code = asyncio.run(_exec_session(ws_url, token, cmd_str))
    sys.exit(exit_code)
